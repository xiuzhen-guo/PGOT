# -*- coding: gbk -*-

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
from networks.HUNet import unet3D_prob
from networks.VNet import VNet_prob
from networks.MCNet import MCNet3d_v1
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
    batch_size, image_h, image_w, image_d = inputs.shape
    inputs = torch.relu(inputs)
    outputs = torch.zeros([batch_size, num_class, image_h, image_w, image_d]).to(inputs.device)
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

        if backbone=='unet':
            self.backbone3d = unet3D_prob(in_channels=self.inchannel, n_classes=self.nclasses, outdim=embed_dim)
        elif backbone=='vnet':
            self.backbone3d = VNet_prob(n_channels=self.inchannel, n_classes=self.nclasses,outdim=embed_dim,
                                        normalization='batchnorm', has_dropout=False) # has_dropout=True
        elif backbone=='mcnet':
            self.backbone3d = MCNet3d_v1(n_channels=self.inchannel, n_classes=self.nclasses, outdim=embed_dim, normalization='batchnorm', has_dropout=False)
                                        
        # initialize after several iterations
        if (proto_mu and proto_sigma) is None:
            self.prototypes_mu = nn.Parameter(torch.zeros(self.nclasses, embed_dim), requires_grad=True)
            self.prototypes_sigma = nn.Parameter(torch.zeros(self.nclasses, embed_dim), requires_grad=True)

    def warm_up(self, x):

        out_seg_1, out_seg_2, representation_1, representation_2 = self.backbone3d(x)
        
        return out_seg_1, out_seg_2, representation_1, representation_2

    def forward(
            self,
            x,
            label=None,
            mode = "train"
    ):
        """
        :param x: size:(B,C,H,W,D)
        :param label: (B,H,W,D)
        :param mask: (B,H,W,D) indicates the result with high confidence, if None, mask equals all ones
        :param update_prototype: whether update the prototype
        """
        B, _, _, _, _,= x.size()

        if label != None:
            label2d = rearrange(label, " b h w d-> (b d) h w")

        return_dict = {}
        out_seg_3d_1, out_seg_3d_2, feature3d_1, feature3d_2 = self.backbone3d(x)
        return_dict["cls_seg_3d_1"] = out_seg_3d_1
        return_dict["cls_seg_3d_2"] = out_seg_3d_2
        return_dict["feature3d_1"] = feature3d_1
        return_dict["feature3d_2"] = feature3d_2

        b,c,h,w,d = feature3d_2.shape

        if mode == "test":
            pass
        else:
            # if prototypes_mu and prototypes_sigma are all zeros, initialize them with current probabilistic embeddings
            tmp = torch.zeros_like(self.prototypes_mu)
            if torch.equal(tmp, self.prototypes_mu) and torch.equal(tmp,self.prototypes_sigma):
                print("Initializing the prototypes!!!!!!!!!!!!")
                flag = self.initialize(feature3d_2, label, self.nclasses) 
                    
            return_dict["proto_mu"] = self.prototypes_mu
            return_dict["proto_sigma"] = self.prototypes_sigma
        
        return return_dict

        
    def _calc_pos(self, select_pos, h, w, d, batch_idx=None):
       
        if isinstance(batch_idx, int):
            batch_idx = torch.full_like(select_pos, batch_idx)
        elif batch_idx is None:
            batch_idx = torch.zeros_like(select_pos)
        
        hwd = h * w * d
        return torch.stack([
            batch_idx,
            (select_pos % hwd) // (w * d),
            (select_pos % (w * d)) // d,
            select_pos % d
        ], dim=1)
       
        
    def initialize(self, ft_3d, label, num_classes, low_ratio=0.2, sigma=0.2):
        
        B, C, H, W, D = ft_3d.shape
        # ===== 1. FFT =====
        fx = torch.fft.fftn(ft_3d.float(), dim=(2,3,4))
        fx = torch.fft.fftshift(fx, dim=(2,3,4))
    
        # ===== 2. 构建高斯频率滤波器 =====
        coords = torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, H),
            torch.linspace(-1, 1, W),
            torch.linspace(-1, 1, D)
        )).to(ft_3d.device)
        dist = coords.norm(dim=0)   # 半径 (H,W,D)
    
        lp = torch.exp(-(dist ** 2) / (2 * sigma ** 2))   # 高斯低通
        hp = 1 - lp                                       # 高斯高通
    
        lp = lp[None, None]   # 广播
        hp = hp[None, None]
    
        # ===== 3. 频域滤波 =====
        low_freq  = fx * lp
        high_freq = fx * hp
    
        # ===== 4. 逆 FFT =====
        l_ft = torch.fft.ifftn(torch.fft.ifftshift(low_freq), dim=(2,3,4)).real
        h_ft = torch.fft.ifftn(torch.fft.ifftshift(high_freq), dim=(2,3,4)).real
        
        l_ft = rearrange(l_ft, "b c h w d-> b h w d c")
        h_ft = rearrange(h_ft, "b c h w d-> b h w d c")
        ft_3d = rearrange(ft_3d, "b c h w d-> b h w d c")
        
        protos_mu_curr = []
        protos_sigma_curr = []        
        for i in range(num_classes):
            if (label==i).sum()>0:
                
                class_l_ft = l_ft[label==i]  # (N, C)
                class_h_ft = h_ft[label==i]
                
                proto_mu = (class_l_ft).mean(dim=0)
                
                diff = class_h_ft - proto_mu     # 利用高频的变化来表征离散性
                var  = (diff ** 2 ).mean(dim=0)
                proto_sigma = torch.sqrt(var + 1e-6)
                
                protos_mu_curr.append(proto_mu.unsqueeze(0))
                protos_sigma_curr.append(proto_sigma.unsqueeze(0))
                del proto_mu, proto_sigma
                
            else: 
                zero_proto = torch.zeros(32, device=l_ft.device)  # (32,)
                protos_mu_curr.append(zero_proto.unsqueeze(0))  # (1, 32)
                protos_sigma_curr.append(zero_proto.unsqueeze(0))  # (1, 32)        
        
        protos_mu_curr = torch.cat(protos_mu_curr, dim=0)  # C, dim
        protos_sigma_curr = torch.cat(protos_sigma_curr, dim=0)

        with torch.no_grad(): 
            self.prototypes_mu.copy_(protos_mu_curr)
            self.prototypes_sigma.copy_(protos_sigma_curr)
            
        return True
        
    def initialize_ablation(self, ft_3d, label, num_classes, low_ratio=0.2):
        
        B, C, H, W, D = ft_3d.shape           
        ft_3d = rearrange(ft_3d, "b c h w d-> b h w d c")
        
        protos_mu_curr = []
        protos_sigma_curr = []        
        for i in range(num_classes):
            if (label==i).sum()>0:
                
                class_ft = ft_3d[label==i]  # (N, C)
                
                proto_mu = (class_ft).mean(dim=0)                           
                proto_sigma = ((class_ft - proto_mu)**2).mean(dim=0).sqrt()
                
                protos_mu_curr.append(proto_mu.unsqueeze(0))
                protos_sigma_curr.append(proto_sigma.unsqueeze(0))
                del proto_mu, proto_sigma
                
            else: 
                zero_proto = torch.zeros(32, device=l_ft.device)  # (32,)
                protos_mu_curr.append(zero_proto.unsqueeze(0))  # (1, 32)
                protos_sigma_curr.append(zero_proto.unsqueeze(0))  # (1, 32)        
        
        protos_mu_curr = torch.cat(protos_mu_curr, dim=0)  # C, dim
        protos_sigma_curr = torch.cat(protos_sigma_curr, dim=0)

        with torch.no_grad(): 
            self.prototypes_mu.copy_(protos_mu_curr)
            self.prototypes_sigma.copy_(protos_sigma_curr)
            
        return True
        
    def sdot_map(self, args, sr_mu, sr_sigma, select_ft, optimization_num, self_snapshot_path):
        
        sdot_maps = get_OT_solver(args, sr_mu, sr_sigma, select_ft, optimization_num, self_snapshot_path)    
        anchor_dir = os.path.join(self_snapshot_path, "features")
        if not os.path.exists(anchor_dir):       
            os.makedirs(anchor_dir)
        save_path = os.path.join(anchor_dir, "anchor_features.pt")
        torch.save({"anchor_features": select_ft,          
                    "shape": select_ft.shape}, save_path)
        
        return sdot_maps
        
        
    def sr_feature_selection_fixed(self, ft_3d, self_snapshot_path):
        
        """
        返回:
            per_class_feats: 每类选中的特征 list[num_classes, (N_c, C)]
            all_classes_mask: [num_classes, B,H,W,D] bool
            all_classes_val:  [num_classes, B,H,W,D] float
        """
        device = ft_3d.device
        B, C, H, W, D = ft_3d.shape
        V = B * H * W * D
    
        # reshape 成 (V, C)
        ft_1d = rearrange(ft_3d, "b c h w d -> (b h w d) c")
    
        chunk = self.config['OT']['bat_size_tg']
        n_chunks = (V + chunk - 1) // chunk
        max_chunks = 2
        
        chunk_ids = torch.randperm(n_chunks, device=device)[:max_chunks]
        print("chunk_ids:", chunk_ids)
        
        num_classes = len(self.prototypes_mu)
    
        # 全局容器
        all_inds = [[] for _ in range(num_classes)]
        all_vals = [[] for _ in range(num_classes)]
        
        for i in chunk_ids.tolist():
            
            s = i * chunk
            e = min((i + 1) * chunk, V)
    
            feats = ft_1d[s:e] 
            print("feats:", feats.shape)
            
            if feats.shape[0] != chunk:
                continue      
    
            if feats.numel() == 0:
                continue
    
            cls_slt = transfer_and_remove_singular_points(
                self.config['OT'], self.prototypes_mu, self.prototypes_sigma, feats, self_snapshot_path
            )
    
            for c in range(num_classes):
                
                inds_local = cls_slt[c]["tg_index"]
                vals_local = cls_slt[c]["weights"]
                if not isinstance(inds_local, torch.Tensor):
                    inds_local = torch.as_tensor(inds_local, dtype=torch.long, device=device)
                    vals_local = torch.as_tensor(vals_local, dtype=torch.long, device=device)
                inds_local = inds_local.to(device)
                vals_local = vals_local.to(device)
    
                if inds_local.numel() == 0:
                    continue
    
                # 转为全局索引
                inds_global = inds_local + s
    
                all_inds[c].append(inds_global)
                all_vals[c].append(vals_local)
    
        per_class_feats = []
        per_class_mask  = []
        per_class_val   = []
    
        for c in range(num_classes):
    
            if len(all_inds[c]) > 0:

                inds = torch.cat(all_inds[c], dim=0)     # (Nc,)
                vals = torch.cat(all_vals[c], dim=0)     # (Nc,)
                vals = torch.softmax(vals, dim=0)
    
                feats = ft_1d[inds]                      # (Nc, C)
    
            else:
                inds = torch.tensor([], device=device, dtype=torch.long)
                vals = torch.tensor([], device=device, dtype=torch.float)
                feats = torch.zeros((0, C), device=device)
    
            mask_flat = torch.zeros(V, dtype=torch.bool, device=device)
            val_flat  = torch.zeros(V, dtype=torch.float, device=device)
    
            mask_flat[inds] = True
            val_flat[inds]  = vals
    
            mask_4d = rearrange(mask_flat, "(b h w d)->b h w d", 
                                b=B, h=H, w=W, d=D)
            val_4d  = rearrange(val_flat, "(b h w d)->b h w d",
                                b=B, h=H, w=W, d=D)
    
            per_class_feats.append(feats)
            per_class_mask.append(mask_4d.unsqueeze(0))   # [1,B,H,W,D]
            per_class_val.append(val_4d.unsqueeze(0))     # [1,B,H,W,D]
    
        # 拼接成 [num_classes, B,H,W,D]
        all_classes_mask = torch.cat(per_class_mask, dim=0)
        all_classes_val  = torch.cat(per_class_val,  dim=0)
    
        return all_classes_mask, all_classes_val
        
    def sr_feature_selection(self, ft_3d, self_snapshot_path):

        device = ft_3d.device
        # reshape 成 (V, C)
        ft_1d = rearrange(ft_3d, "b c h w d -> (b h w d) c")     
    
        select_unabeled_features, confidence_scores = semi_discrete_OT_selecting_features(
            self.config['OT'], self.prototypes_mu, self.prototypes_sigma, ft_1d, self_snapshot_path)
    
        return select_unabeled_features, confidence_scores

   
    def prototype_update(self, ft_3d, label, num_classes, mask=None, confidence=None):
        
        b,c,h,w,d = ft_3d.shape
        label_onehot = label_to_onehot(label, num_class=self.nclasses)
        if mask != None:
            mask = mask
        else:
            mask = torch.ones((b, h, w, d)).cuda()

        return self.prototype_learning(ft_3d, label, mask, confidence, num_classes)
    
    def prototype_learning_ablation(self, ft_3d, label, mask, confidence, num_classes, low_ratio=0.2):
        
        device = ft_3d.device
        B, C, H, W, D = ft_3d.shape
        ft_3d = rearrange(ft_3d, "b c h w d-> b h w d c")

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
                        proto_sigma_ = torch.full((1, protos_mu_prev.size(-1)), 1e+32).cuda()
                        proto_mu_ = torch.zeros((1, protos_mu_prev.size(-1))).cuda()
                else:        
                    with torch.no_grad():                                                                                
                        
                        class_conf = confidence[i][mask_bool[i]].unsqueeze(1)
                        class_ft = ft_3d[mask_bool[i]]  # (N, C)
                        
                        proto_mu_ = (class_ft * class_conf).sum(dim=0) / class_conf.sum(dim=0)                                   
                        proto_sigma_ = (class_conf*(class_ft - proto_mu_)**2).sum(dim=0)/class_conf.sum(dim=0).sqrt()                                                                                                      
            else:
                proto_sigma_=proto_mu_=torch.zeros(32, device=l_ft.device)        
                                       
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
            
    def prototype_learning(self, ft_3d, label, mask, confidence, num_classes, low_ratio=0.2, sigma=0.2):
        
        device = ft_3d.device
        B, C, H, W, D = ft_3d.shape
        
        # 空间多尺度
        #low = F.avg_pool3d(ft_3d, kernel_size=3, stride=1, padding=1)
        #high = ft_3d - low
        
        B, C, H, W, D = ft_3d.shape
        # ===== 1. FFT =====
        fx = torch.fft.fftn(ft_3d.float(), dim=(2,3,4))
        fx = torch.fft.fftshift(fx, dim=(2,3,4))
    
        # ===== 2. 构建高斯频率滤波器 =====
        coords = torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, H),
            torch.linspace(-1, 1, W),
            torch.linspace(-1, 1, D)
        )).to(ft_3d.device)
        dist = coords.norm(dim=0)   # 半径 (H,W,D)
    
        lp = torch.exp(-(dist ** 2) / (2 * sigma ** 2))   # 高斯低通
        hp = 1 - lp                                       # 高斯高通
    
        lp = lp[None, None]   # 广播
        hp = hp[None, None]
    
        # ===== 3. 频域滤波 =====
        low_freq  = fx * lp
        high_freq = fx * hp
    
        # ===== 4. 逆 FFT =====
        l_ft = torch.fft.ifftn(torch.fft.ifftshift(low_freq), dim=(2,3,4)).real
        h_ft = torch.fft.ifftn(torch.fft.ifftshift(high_freq), dim=(2,3,4)).real
        
        l_ft = rearrange(l_ft, "b c h w d-> b h w d c")
        h_ft = rearrange(h_ft, "b c h w d-> b h w d c")
        ft_3d = rearrange(ft_3d, "b c h w d-> b h w d c")
        
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
                        proto_sigma_ = torch.full((1, protos_mu_prev.size(-1)), 1e+32).cuda()
                        proto_mu_ = torch.zeros((1, protos_mu_prev.size(-1))).cuda()
                else:        
                    with torch.no_grad():                                                                                
                        
                        class_conf = confidence[i][mask_bool[i]].unsqueeze(1)
                        class_l_ft = l_ft[mask_bool[i]]  # (N, C)
                        class_h_ft = h_ft[mask_bool[i]]        
                                                
                        weights = class_conf
                        
                        proto_mu_ = (class_l_ft * weights).sum(dim=0) / weights.sum(dim=0)  
                        
                        diff = class_h_ft - proto_mu_     # 利用高频的变化来表征离散性
                        var  = (diff ** 2 * weights).sum(dim=0) / weights.sum(dim=0)
                        proto_sigma_ = torch.sqrt(var + 1e-6)
                                                                                                                               
            else:
                proto_sigma_=proto_mu_=torch.zeros(32, device=l_ft.device)        
                                       
            protos_mu_curr.append(proto_mu_.unsqueeze(0))
            protos_sigma_curr.append(proto_sigma_.unsqueeze(0))
        
        protos_mu_curr = torch.cat(protos_mu_curr, dim=0) 
        protos_sigma_curr = torch.cat(protos_sigma_curr, dim=0)

        # Prototype updating
        if protos_sigma_prev.sum() !=0 and protos_sigma_curr.sum() !=0:
            protos_sigma_new = 1 / torch.add(1 / protos_sigma_prev, 1/protos_sigma_curr)
            protos_mu_new = torch.add((protos_sigma_new / protos_sigma_prev) * protos_mu_prev,
                                     (protos_sigma_new / protos_sigma_curr) * protos_mu_curr)
            print("class:", i)
            print(torch.isnan(protos_mu_new).any(), protos_mu_new.min(), protos_mu_new.max(),
                      torch.isnan(protos_sigma_new).any(),protos_sigma_new.min(), protos_sigma_new.max())
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
        
    def prototype_learning_brats2024(
        self, ft_3d, label, mask, confidence,
        num_classes, sigma=0.2, eps=1e-6
    ):
        device = ft_3d.device
        B, C, H, W, D = ft_3d.shape
    
        # ================= FFT =================
        fx = torch.fft.fftn(ft_3d.float(), dim=(2, 3, 4))
        fx = torch.fft.fftshift(fx, dim=(2, 3, 4))
    
        coords = torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            torch.linspace(-1, 1, D, device=device),
            indexing="ij"
        ))
        dist = coords.norm(dim=0)
    
        lp = torch.exp(-(dist ** 2) / (2 * sigma ** 2))
        hp = 1.0 - lp
        lp = lp[None, None]
        hp = hp[None, None]
    
        low_freq = fx * lp
        high_freq = fx * hp
    
        l_ft = torch.fft.ifftn(torch.fft.ifftshift(low_freq), dim=(2, 3, 4)).real
        h_ft = torch.fft.ifftn(torch.fft.ifftshift(high_freq), dim=(2, 3, 4)).real
    
        l_ft = rearrange(l_ft, "b c h w d -> b h w d c")
        h_ft = rearrange(h_ft, "b c h w d -> b h w d c")
    
        protos_mu_prev = self.prototypes_mu.detach()
        protos_sigma_prev = self.prototypes_sigma.detach()
    
        protos_mu_curr = []
        protos_sigma_curr = []
    
        mask_bool = mask.bool()
    
        for cls in range(num_classes):
    
            # ---------- 类别完全缺失 ----------
            if (label == cls).sum() == 0 or mask_bool[cls].sum() == 0:
                protos_mu_curr.append(protos_mu_prev[cls:cls+1])
                protos_sigma_curr.append(protos_sigma_prev[cls:cls+1])
                continue
    
            class_l = l_ft[mask_bool[cls]]
            class_h = h_ft[mask_bool[cls]]
            weights = confidence[cls][mask_bool[cls]].unsqueeze(1)
    
            w_sum = weights.sum().clamp(min=eps)
    
            # ---------- 均值（低频） ----------
            proto_mu = (class_l * weights).sum(dim=0) / w_sum
    
            # ---------- 方差（高频） ----------
            diff = class_h - proto_mu
            var = (diff ** 2 * weights).sum(dim=0) / w_sum
            proto_sigma = torch.sqrt(var + eps)
    
            protos_mu_curr.append(proto_mu.unsqueeze(0))
            protos_sigma_curr.append(proto_sigma.unsqueeze(0))
    
        protos_mu_curr = torch.cat(protos_mu_curr, dim=0)
        protos_sigma_curr = torch.cat(protos_sigma_curr, dim=0)
    
        # ================= 融合更新 =================
        valid_prev = protos_sigma_prev > eps
        valid_curr = protos_sigma_curr > eps
        valid = valid_prev & valid_curr
    
        protos_sigma_new = protos_sigma_prev.clone()
        protos_mu_new = protos_mu_prev.clone()
    
        denom = (1.0 / protos_sigma_prev + 1.0 / protos_sigma_curr).clamp(min=eps)
        protos_sigma_new[valid] = 1.0 / denom[valid]
    
        protos_mu_new[valid] = (
            protos_sigma_new[valid] / protos_sigma_prev[valid] * protos_mu_prev[valid] +
            protos_sigma_new[valid] / protos_sigma_curr[valid] * protos_mu_curr[valid]
        )
    
        # ================= 写回 =================
        with torch.no_grad():
            self.prototypes_mu.copy_(protos_mu_new)
            self.prototypes_sigma.copy_(protos_sigma_new)
    
        # ================= 调试检查 =================
        assert not torch.isnan(self.prototypes_mu).any(), "NaN in prototype mu"
        assert not torch.isnan(self.prototypes_sigma).any(), "NaN in prototype sigma"
    
        proportion = (
            protos_sigma_curr.mean(-1) /
            protos_sigma_prev.mean(-1).clamp(min=eps)
        ).max()
    
        return proportion

        

    def sdot_score(self, args, mu, sigma, feature3d, optimization_num, self_snapshot_path):
          
        device = feature3d.device
        probabilities, pseudo_labels = OT_Map(args, mu, sigma, feature3d, optimization_num, self_snapshot_path)  
        
        return probabilities.to(device), pseudo_labels.to(device)

