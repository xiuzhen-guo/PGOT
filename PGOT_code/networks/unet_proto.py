import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import os
import yaml
import pywt
import numpy as np
import torch.fft as fft
import sys
sys.path.insert(0, "../../")
from networks.unetmodel import UNet
from ot_utils.ot_util import get_OT_solver, OT_Map, transfer_and_remove_singular_points

    
def momentum_update(old_mu, new_mu, old_sigma, new_sigma, momentum):
    update_mu = momentum * old_mu + (1 - momentum) * new_mu
    update_sigma = momentum * old_sigma + (1 - momentum) * new_sigma
    return update_mu, update_sigma


def label_to_onehot(inputs, num_class):
    '''
    inputs is class label
    return one_hot label
    dim will be increasee
    '''
    batch_size, image_h, image_w = inputs.shape
    inputs = torch.relu(inputs)
    outputs = torch.zeros([batch_size, num_class, image_h, image_w]).to(inputs.device)
    return outputs.scatter_(1, inputs.unsqueeze(1), 1.0)


class UNetProto(nn.Module):
    def __init__(
            self,
            config,
            backbone,
            inchannel,
            nclasses,
            proto_mu=None,
            proto_sigma=None,
            embed_dim=256,
            momentum=False

    ):
        super().__init__()
        self.inchannel = inchannel
        self.nclasses = nclasses
        self.momentum = momentum
        with open(os.path.join('/home/shiji/gxz/code/PG-OT/code/configs/', config), 'r') as f:
            self.config = yaml.safe_load(f)

        self.backbone = UNet(self.inchannel, self.nclasses, out_dim=embed_dim)
                                        
        # initialize after several iterations
        if (proto_mu and proto_sigma) is None:
            self.prototypes_mu = nn.Parameter(torch.zeros(self.nclasses, embed_dim), requires_grad=True)
            self.prototypes_sigma = nn.Parameter(torch.zeros(self.nclasses, embed_dim), requires_grad=True)

    def warm_up(self, x):

        out_seg, representation = self.backbone(x)
        
        return out_seg, representation

    def forward(
            self,
            x,
            label=None,
            mode = "train"
    ):
        """
        :param x: size:(B,C,H,W)
        :param label: (B,H,W)
        :param update_prototype: whether update the prototype
        """
        B, _, _, _, = x.size()

        return_dict = {}
        out_seg_2d, feature2d = self.backbone(x)
        return_dict["cls_seg_2d"] = out_seg_2d
        return_dict["feature2d"] = feature2d
        b,c,h,w = feature2d.shape

        if mode == "test":
            pass
        else:
            # if prototypes_mu and prototypes_sigma are all zeros, initialize them with current probabilistic embeddings
            tmp = torch.zeros_like(self.prototypes_mu)
            if torch.equal(tmp, self.prototypes_mu) and torch.equal(tmp,self.prototypes_sigma):
                print("Initializing the prototypes!!!!!!!!!!!!")
                flag = self.initialize(feature2d, label, self.nclasses) 
                    
            return_dict["proto_mu"] = self.prototypes_mu
            return_dict["proto_sigma"] = self.prototypes_sigma
        
        return return_dict
        
    
    def tg_feature_selection(self, ft_2d):
    
        '''
        采样bat_size_tg个像素特征，用于训练sdot
        '''
        
        b, c, h, w = ft_2d.shape

        if b <= 2:
            ft_1d = rearrange(ft_2d, "b c h w -> (b h w) c")
            select_pos = torch.randperm(ft_1d.shape[0])[:self.config['OT']['bat_size_tg']]
            select_ft = ft_1d[select_pos]
            
            b_cods = select_pos // (h * w)
            remaining = select_pos % (h * w)
            h_cods = remaining // w
            remaining = remaining % w
            w_cods = remaining % d
            pos = torch.stack([b_cods, h_cods, w_cods], dim=1)  
            
        else:
            ft_lab = rearrange(ft_2d[:2], "b c h w -> b (h w) c")  
            ft_unlab = rearrange(ft_2d[2:], "b c h w -> (b h w) c") 
            
            num_1 = self.config['OT']['bat_size_tg']//4
            select_ft_lab = torch.cat([
                ft_lab[i][torch.randperm(ft_lab.shape[1])[:num_1]] 
                for i in range(2)
            ], dim=0)  
            
            num_2 = self.config['OT']['bat_size_tg']//2
            select_ft_unlab = ft_unlab[torch.randperm(ft_unlab.shape[0])[:num_2]]  
            select_ft = torch.cat([select_ft_lab, select_ft_unlab], dim=0) 
            
            pos_lab = torch.cat([
                self._calc_pos(torch.randperm(h*w)[:num_1], h, w, batch_idx=i)
                for i in range(2)
            ], dim=0)
            
            pos_unlab = self._calc_pos(
                torch.randperm(ft_unlab.shape[0])[:num_2] % (h*w), h, w,
                batch_idx=(torch.randperm(ft_unlab.shape[0])[:num_2] // (h*w)) + 2)
            pos = torch.cat([pos_lab, pos_unlab], dim=0)    
        
        return select_ft, pos
        
        
    def _calc_pos(self, select_pos, h, w, batch_idx=None):
       
        if isinstance(batch_idx, int):
            batch_idx = torch.full_like(select_pos, batch_idx)
        elif batch_idx is None:
            batch_idx = torch.zeros_like(select_pos)
        
        hwd = h * w
        return torch.stack([
            batch_idx,
            (select_pos % (h * w)) // w,
            select_pos % w
        ], dim=1)
        
    def initialize(self, ft_2d, label, num_classes, low_ratio=0.2):
        
        B, C, H, W = ft_2d.shape
        fx = torch.fft.fftn(ft_2d.float(), dim=(2,3))
        fx = torch.fft.fftshift(fx, dim=(2,3))
        
        freqs = torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, H),
            torch.linspace(-1, 1, W)
        )).norm(dim=0).to(ft_2d.device)  
        
        _, idx = torch.sort(freqs.flatten())
        low_mask = torch.zeros_like(freqs)
        low_mask.view(-1)[idx[:int(low_ratio*H*W)]] = 1
        
        low_freq = fx * low_mask
        high_freq = fx * (1 - low_mask)
        
        l_ft = torch.fft.ifftn(torch.fft.ifftshift(low_freq)).real
        h_ft = torch.fft.ifftn(torch.fft.ifftshift(high_freq)).real
        
        l_ft = rearrange(l_ft, "b c h w -> b h w c")
        h_ft = rearrange(h_ft, "b c h w -> b h w c")
        ft_2d = rearrange(ft_2d, "b c h w-> b h w c")
        
        protos_mu_curr = []
        protos_sigma_curr = []        
        for i in range(num_classes):
            if (label==i).sum()>0:
                
                class_l_ft = l_ft[label==i]  # (N, C)
                class_h_ft = h_ft[label==i]
                
                l_energy = class_l_ft.pow(2).mean(dim=0, keepdim=True)  # (1, C)
                h_energy = class_h_ft.pow(2).mean(dim=0, keepdim=True)
                
                alpha = 0.7  
                l_weights = alpha * l_energy / (l_energy + h_energy + 1e-6)
                h_weights = (1-alpha) * h_energy / (l_energy + h_energy + 1e-6)
                
                proto_mu = (class_l_ft * l_weights).sum(dim=0) / (l_weights.sum() + 1e-6)
                proto_sigma = ((class_h_ft - proto_mu).pow(2) * h_weights).sum(dim=0).sqrt() / (h_weights.sum() + 1e-6)    
                
                protos_mu_curr.append(proto_mu.unsqueeze(0))
                protos_sigma_curr.append(proto_sigma.unsqueeze(0))
                del proto_mu, proto_sigma
            else: 
                zero_proto = torch.zeros(embed_dim, device=l_ft.device)  # (32,)
                protos_mu_curr.append(zero_proto.unsqueeze(0))  # (1, 32)
                protos_sigma_curr.append(zero_proto.unsqueeze(0))  # (1, 32)        
        
        protos_mu_curr = torch.cat(protos_mu_curr, dim=0)  # C, dim
        protos_sigma_curr = torch.cat(protos_sigma_curr, dim=0)

        with torch.no_grad(): 
            self.prototypes_mu.copy_(protos_mu_curr)
            self.prototypes_sigma.copy_(protos_sigma_curr)
            
        return True
        
    def sdot_map(self, args, sr_mu, sr_sigma, select_ft):
        
        sdot_maps = get_OT_solver(args, sr_mu, sr_sigma, select_ft)    
        
        return sdot_maps
        
    def sr_feature_selection(self, ft_2d, label):
        
        """
        用于更新原型的特征需要过滤奇异点
        """
        
        device = ft_2d.device
        B, C, H, W = ft_2d.shape
        
        ft_1d = rearrange(ft_2d, "b c h w -> (b h w) c")  
        
        chunk_size = self.config['OT']['bat_size_tg']
        total_voxels = B * H * W
        num_chunks = (total_voxels + chunk_size - 1) // chunk_size
        
        all_class_indices = [[] for _ in range(len(self.prototypes_mu))]
        
        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size
            end = min((chunk_idx + 1) * chunk_size, total_voxels)
            
            chunk_feats = ft_1d[start:end]
            if len(chunk_feats) == 0:
                continue
                
            _, chunk_slt_inds = transfer_and_remove_singular_points(
                self.config['OT'],
                self.prototypes_mu,
                self.prototypes_sigma,
                chunk_feats
            )
            
            if not chunk_slt_inds:
                continue
                
            for class_idx in range(len(chunk_slt_inds)):
                if len(chunk_slt_inds[class_idx]) > 0:
                    local_indices = chunk_slt_inds[class_idx]
                    global_indices = start + local_indices.int()
                    all_class_indices[class_idx].append(global_indices)
        
        all_class_indices = [
            torch.cat(indices) if indices else torch.tensor([], dtype=torch.long, device=device)
            for indices in all_class_indices
        ]
        
        # 计算非空类别
        valid_classes = [i for i, indices in enumerate(all_class_indices) if len(indices) > 0]
        if len(valid_classes) == 0:
            spatial_mask = torch.zeros(B, H, W, dtype=torch.bool, device=device)
            return torch.zeros(0, C, device=device), spatial_mask
        
        # 计算交集（使用numpy.intersect1d）
        common_indices = all_class_indices[valid_classes[0]].cpu().numpy()
        for i in range(1, len(valid_classes)):
            current_indices = all_class_indices[valid_classes[i]].cpu().numpy()
            common_indices = np.intersect1d(common_indices, current_indices)
        
        common_indices = torch.tensor(common_indices, device=device)
        
        # 创建mask
        spatial_mask = torch.zeros(B * H * W, dtype=torch.bool, device=device)
        if len(common_indices) > 0:
            spatial_mask[common_indices] = 1
        spatial_mask = rearrange(spatial_mask, "(b h w) -> b h w", b=B, h=H, w=W)
        
        # 提取特征
        common_feats = ft_1d[common_indices] if len(common_indices) > 0 else torch.zeros(0, C, device=device)
        
        return common_feats, spatial_mask
        
        
    def prototype_update(self, ft_2d, label, num_classes, mask=None):
        
        b,c,h,w = ft_2d.shape

        label_onehot = label_to_onehot(label, num_class=self.nclasses)
        if mask != None:
            mask = mask
        else:
            mask = torch.ones((b, h, w)).cuda()

        return self.prototype_learning(ft_2d, label, mask, num_classes)
    
    def prototype_learning(self, ft_2d, label, mask, num_classes, low_ratio=0.2):
        
        device = ft_2d.device
        B, C, H, W = ft_2d.shape
        fx = torch.fft.fftn(ft_2d.float(), dim=(2,3))
        fx = torch.fft.fftshift(fx, dim=(2,3))
        
        freqs = torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, H),
            torch.linspace(-1, 1, W)
        )).norm(dim=0).to(ft_2d.device)  # (H,W)
        
        _, idx = torch.sort(freqs.flatten())
        low_mask = torch.zeros_like(freqs)
        low_mask.view(-1)[idx[:int(low_ratio*H*W)]] = 1
        
        low_freq = fx * low_mask
        high_freq = fx * (1 - low_mask)
        
        l_ft = torch.fft.ifftn(torch.fft.ifftshift(low_freq)).real
        h_ft = torch.fft.ifftn(torch.fft.ifftshift(high_freq)).real
        
        l_ft = rearrange(l_ft, "b c h w -> b h w c")
        h_ft = rearrange(h_ft, "b c h w -> b h w c")
        ft_2d = rearrange(ft_2d, "b c h w -> b h w c")

        num_segments = num_classes
        protos_mu_prev = self.prototypes_mu
        protos_sigma_prev = self.prototypes_sigma
        protos_mu_curr = []
        protos_sigma_curr = []
        mask_bool = mask.bool()
        for i in range(num_segments):  
            if (label==i).sum()>0:
                valid_pixel = mask
                if valid_pixel.sum() == 0:
                    if self.momentum:
                        proto_sigma_ = protos_sigma_prev[i].unsqueeze(0)
                        proto_mu_ = protos_mu_prev[i].unsqueeze(0)
                    else:
                        proto_sigma_ = torch.full((protos_mu_prev.size(-1),), 1e+32).cuda()
                        proto_mu_ = torch.zeros(protos_mu_prev.size(-1)).cuda()
                        
                else:        
                    with torch.no_grad():                    
                                             
                        class_l_ft = l_ft[mask_bool][label[mask_bool] == i]  # (N, C)
                        class_h_ft = h_ft[mask_bool][label[mask_bool] == i]
                        
                        l_energy = class_l_ft.pow(2).mean(dim=0, keepdim=True)  # (1, C)
                        h_energy = class_h_ft.pow(2).mean(dim=0, keepdim=True)
                        
                        alpha = 0.7  
                        l_weights = alpha * l_energy / (l_energy + h_energy + 1e-6)
                        h_weights = (1-alpha) * h_energy / (l_energy + h_energy + 1e-6)
                        
                        proto_mu_ = (class_l_ft * l_weights).sum(dim=0) / (l_weights.sum() + 1e-6)
                        proto_sigma_ = ((class_h_ft - proto_mu_).pow(2) * h_weights).sum(dim=0).sqrt() / (h_weights.sum() + 1e-6)  
            else:
                proto_sigma_=proto_mu_=torch.zeros(embed_dim, device=l_ft.device)        
                                       
            protos_mu_curr.append(proto_mu_.unsqueeze(0))
            protos_sigma_curr.append(proto_sigma_.unsqueeze(0))  
        
        protos_mu_curr = torch.cat(protos_mu_curr, dim=0) 
        protos_sigma_curr = torch.cat(protos_sigma_curr, dim=0)

        # Prototype updating
        if protos_sigma_prev.sum() !=0 and protos_sigma_curr.sum() !=0:
            protos_sigma_new = 1 / torch.add(1 / protos_sigma_prev, 1/protos_sigma_curr)
            protos_mu_new = torch.add((protos_sigma_new / protos_sigma_prev) * protos_mu_prev,
                                     (protos_sigma_new / protos_sigma_curr) * protos_mu_curr)
        elif protos_sigma_prev.sum() !=0:
            protos_sigma_new = protos_sigma_prev
            protos_mu_new = protos_sigma_prev
        else:
            protos_sigma_new = protos_sigma_curr
            protos_mu_new = protos_sigma_curr            

        with torch.no_grad():
            self.prototypes_mu.copy_(protos_mu_new.clone())
            self.prototypes_sigma.copy_(protos_sigma_new.clone())

        # calculate the proportion of curr information and prev
        proportion = torch.max(protos_sigma_curr.mean(-1) / protos_sigma_prev.mean(-1))
        
        return proportion
        

    def sdot_score(self, args, mu, sigma, ft_2d):
        """
        随机采样若干个 self.config['OT']['bat_size_tg'] 大小的特征块计算 OT 相似度，近似估计全局相似度
        Args:
            args: 参数配置
            mu: 源原型均值 (num_classes, C)
            sigma: 源原型方差 (num_classes, C)
            ft_2d: 目标特征 (B, C, H, W)
            num_samples: 随机采样多少个 self.config['OT']['bat_size_tg'] 大小的块
        Returns:
            sdot_sims: 近似相似度图 (B, num_classes, H, W)
        """
        device = ft_2d.device
        B, C, H, W = ft_2d.shape
        num_classes = len(mu)
        
        ft_1d = rearrange(ft_2d, "b c h w -> (b h w) c")
        total_pixels = ft_1d.shape[0]
        
        if total_pixels % self.config['OT']['bat_size_tg'] != 0:
            pad_size = self.config['OT']['bat_size_tg'] - (total_pixels % self.config['OT']['bat_size_tg'])
            ft_1d = torch.cat([ft_1d, torch.zeros(pad_size, C, device=device)], dim=0)
            total_pixels = ft_1d.shape[0]
        
        # 3. 随机选择 num_samples 个 self.config['OT']['bat_size_tg'] 大小的块
        chunk_size = self.config['OT']['bat_size_tg']
        num_chunks = total_pixels // chunk_size
        
        # 4. 计算选中的块的 OT 相似度
        sdot_1d = torch.zeros(total_pixels, num_classes, device=device) 
        mask_1d = torch.zeros(total_pixels, dtype=torch.bool, device=device)  
        for idx in range(num_chunks):
            start = idx * chunk_size
            end = start + chunk_size
            chunk_feat = ft_1d[start:end]
            chunk_sims = OT_Map(args, mu, sigma, chunk_feat)  # (chunk_size, num_classes)
            sdot_1d[start:end] = torch.stack(chunk_sims, dim=1).to(device)  
            mask_1d[start:end] = 1  
        
        # 5. 恢复原始形状
        sdot_1d = sdot_1d[:B*H*W]  
        mask_1d = mask_1d[:B*H*W]  
        sdot_sims = rearrange(sdot_1d, "(b h w) k -> b h w k", b=B, h=H, w=W, k=num_classes)
        mask = rearrange(mask_1d, "(b h w) -> b h w", b=B, h=H, w=W)
        
        return sdot_sims, mask  



