import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import math
from torch.quasirandom import SobolEngine

torch.set_printoptions(precision=8)
class OptimalTransport():	
    def __init__ (self, mu, sigma, target_feature, args, device='cuda:0', out_dir='./results/ot_models/'):
        self.mu = mu
        self.device = device
        self.sigma = sigma
        self.tg_fea = target_feature
        self.num_tg = self.tg_fea.shape[0]
        self.dim = self.tg_fea.shape[1]
        self.max_iter = args['max_iter']
        self.lr = args['lr_ot']
        self.bat_size_sr = args['bat_size_sr']
        self.bat_size_tg = args['bat_size_tg']
           
        if self.num_tg % args['bat_size_tg'] != 0:
          sys.exit('Error: (num_tg) is not a multiple of (bat_size_tg)')
        
        self.epochs_per_save = 500
        self.out_dir = out_dir

        self.num_bat_sr = args['num_sr'] // args['bat_size_sr']
        self.num_bat_tg = self.num_tg // args['bat_size_tg']
        #!internal variables
        self.device = device
        self.d_h = torch.zeros(self.num_tg, dtype=torch.float, device=self.device)
        self.d_g = torch.zeros(self.num_tg, dtype=torch.float, device=self.device)
        self.d_g_sum = torch.zeros(self.num_tg, dtype=torch.float, device=self.device)   
    
    def cal_measure_one_batch(self):
        
        '''Calculate the pushed-forward measure of current step. 
        '''
        d_volP_list = []
        for j in range(num_class):
            d_volP_j = torch.randn(self.bat_size_sr // num_class, self.dim, device=self.device) * self.sigma[j] + self.mu[j]
            d_volP_list.append(d_volP_j) 
        d_volP = torch.cat(d_volP_list, dim=0)
        
        prototype_dir = os.path.join(self.out_dir, "features")
        if not os.path.exists(prototype_dir):       
            os.makedirs(prototype_dir)
        save_path = os.path.join(prototype_dir, "prototypes.pt")
        torch.save({"prototypes": d_volP,          
                    "shape": d_volP.shape}, save_path)
        
        d_tot_ind = torch.empty(self.bat_size_sr, dtype=torch.long, device=self.device)
        d_tot_ind_val = torch.empty(self.bat_size_sr, dtype=torch.float, device=self.device)   
        d_tot_ind_val.fill_(-1e8)
        d_tot_ind.fill_(-1)
   
        i = 0 
        while i < self.num_bat_tg:
            temp_tg = self.tg_fea[i*self.bat_size_tg:(i+1)*self.bat_size_tg]
            temp_tg = temp_tg.view(temp_tg.shape[0], -1)	

            '''U=PX+H'''
            d_temp_h = self.d_h[i*self.bat_size_tg:(i+1)*self.bat_size_tg]
            #with torch.no_grad():  
            d_U = torch.mm(temp_tg.float(), d_volP.t().float())+ d_temp_h.expand([self.bat_size_sr, -1]).t()
            '''compute max'''
            d_ind_val, d_ind = torch.max(d_U, 0)
            '''add P id offset'''
            d_ind = d_ind+(i*self.bat_size_tg)
            '''store best value'''
            d_tot_ind_val, d_ind_val_argmax = torch.max(torch.stack([d_tot_ind_val, d_ind_val],dim = 0), 0)
            d_tot_ind = torch.stack([d_tot_ind, d_ind],dim = 0)[d_ind_val_argmax, torch.arange(self.bat_size_sr)]
            '''add step'''
            i = i+1
            #del d_volP, d_U  
        '''calculate histogram'''
        self.d_g.copy_(torch.bincount(d_tot_ind, minlength=self.num_tg))        
      
        
    def cal_measure(self):
        
        self.d_g_sum.fill_(0)
        for count in range(self.num_bat_sr):       
            self.cal_measure_one_batch()
            self.d_g_sum = self.d_g_sum + self.d_g
        self.d_g = self.d_g_sum/(self.num_bat_sr*self.bat_size_sr)
    
        
    def forward(self):
        sr_feature = torch.randn(self.bat_size_sr, self.dim, device=self.device) * self.sigma + self.mu
        ind_len = sr_feature.shape[0]
        d_volP = sr_feature.view(ind_len, -1)
        d_tot_ind = torch.empty(ind_len, dtype=torch.long, device=self.device)
        d_tot_ind_val = torch.empty(ind_len, dtype=torch.float, device=self.device)   
        d_tot_ind_val.fill_(-1e30)
        d_tot_ind.fill_(-1)
        i = 0 
        for i in range(self.num_bat_tg):
            temp_tg = self.tg_fea[i*self.bat_size_tg:(i+1)*self.bat_size_tg]
            temp_tg = temp_tg.view(temp_tg.shape[0], -1)	

            '''U=PX+H'''
            d_temp_h = self.d_h[i*self.bat_size_tg:(i+1)*self.bat_size_tg]
            d_U = torch.mm(temp_tg, d_volP.t()) + d_temp_h.expand([ind_len, -1]).t()
            '''compute max'''
            d_ind_val, d_ind = torch.max(d_U, 0)
            '''add P id offset'''
            d_ind = d_ind+(i*self.bat_size_tg)
            '''store best value'''
            d_tot_ind_val, d_ind_val_argmax = torch.max(torch.stack([d_tot_ind_val, d_ind_val],dim = 0), 0)
            d_tot_ind = torch.stack([d_tot_ind, d_ind],dim = 0)[d_ind_val_argmax, torch.arange(ind_len)]
            '''add step'''
            i = i+1
        return self.tg_fea[d_tot_ind,:,:,:]
        
    def transfer_reliable_feature(self, unlab_ft_1d, w=0.5):

        num_class = self.mu.shape[0]
        d_volP_list = []
        for j in range(num_class):
            d_volP_j = torch.randn(self.bat_size_sr // num_class, self.dim, device=self.device) * self.sigma[j] + self.mu[j]
            d_volP_list.append(d_volP_j)
        sr_feature = torch.cat(d_volP_list, dim=0)
        ind_len = sr_feature.shape[0]
        d_volP = sr_feature.view(ind_len, -1)
    
        per_class_selected_idx = {c: [] for c in range(num_class)}     # tg index
        per_class_selected_w   = {c: [] for c in range(num_class)}     # weight
    
        sr_class_id = torch.arange(num_class, device=self.device).repeat_interleave(self.bat_size_sr // num_class)
        for i in range(self.num_bat_tg):
            temp_tg = self.tg_fea[i*self.bat_size_tg:(i+1)*self.bat_size_tg].view(self.bat_size_tg, -1)
            d_temp_h = self.d_h[i*self.bat_size_tg:(i+1)*self.bat_size_tg]
            d_U = torch.mm(temp_tg, d_volP.t()) + d_temp_h.expand([ind_len, -1]).t()
            d_ind_val, d_ind = torch.max(d_U, dim=0)  
    
            for tg_idx in range(i*self.bat_size_tg, (i+1)*self.bat_size_tg):
    
                hit_sr = (d_ind == tg_idx - i*self.bat_size_tg)
                if hit_sr.sum() == 0:
                    continue  
    
                class_counts = torch.bincount(sr_class_id[hit_sr], minlength=num_class)    
                pseudo_label = class_counts.argmax().item()    
                pseudo_weight = class_counts[pseudo_label].float() / hit_sr.sum().float()
                per_class_selected_idx[pseudo_label].append(tg_idx)
                per_class_selected_w[pseudo_label].append(pseudo_weight)
    
        final_result = {}
        for c in range(num_class):
        
            if len(per_class_selected_idx[c]) == 0:
                final_result[c] = {
                    "tg_index": [],
                    "weights": [],
                }
                continue
        
            idxs  = torch.tensor(per_class_selected_idx[c], device=self.device)
            ws    = torch.tensor(per_class_selected_w[c],  device=self.device)
        
            sorted_w, order = torch.sort(ws, descending=True)
            sorted_idxs = idxs[order]
            mask = sorted_w > w
        
            if mask.sum() == 0:   
                final_result[c] = {
                    "tg_index": [],
                    "weights": [],
                }
                continue
        
            filtered_idxs = sorted_idxs[mask]
            filtered_w    = sorted_w[mask]
            final_result[c] = {
                "tg_index": filtered_idxs,
                "weights":  filtered_w,
            }
        
        return final_result
        
    
    def transfer_topk(self, w=0.5):

        num_class = self.mu.shape[0]
        d_volP_list = []
        for j in range(num_class):
            d_volP_j = torch.randn(self.bat_size_sr // num_class, self.dim, device=self.device) * self.sigma[j] + self.mu[j]
            d_volP_list.append(d_volP_j)
        sr_feature = torch.cat(d_volP_list, dim=0)
        ind_len = sr_feature.shape[0]
        d_volP = sr_feature.view(ind_len, -1)
    
        per_class_selected_idx = {c: [] for c in range(num_class)}     # tg index
        per_class_selected_w   = {c: [] for c in range(num_class)}     # weight
    
        sr_class_id = torch.arange(num_class, device=self.device).repeat_interleave(self.bat_size_sr // num_class)
        for i in range(self.num_bat_tg):
            temp_tg = self.tg_fea[i*self.bat_size_tg:(i+1)*self.bat_size_tg].view(self.bat_size_tg, -1)
            d_temp_h = self.d_h[i*self.bat_size_tg:(i+1)*self.bat_size_tg]
            d_U = torch.mm(temp_tg, d_volP.t()) + d_temp_h.expand([ind_len, -1]).t()
            d_ind_val, d_ind = torch.max(d_U, dim=0)  
    
            for tg_idx in range(i*self.bat_size_tg, (i+1)*self.bat_size_tg):
    
                hit_sr = (d_ind == tg_idx - i*self.bat_size_tg)
                if hit_sr.sum() == 0:
                    continue  
    
                class_counts = torch.bincount(sr_class_id[hit_sr], minlength=num_class)    
                pseudo_label = class_counts.argmax().item()    
                pseudo_weight = class_counts[pseudo_label].float() / hit_sr.sum().float()
                per_class_selected_idx[pseudo_label].append(tg_idx)
                per_class_selected_w[pseudo_label].append(pseudo_weight)
    
        final_result = {}
        for c in range(num_class):
        
            if len(per_class_selected_idx[c]) == 0:
                final_result[c] = {
                    "tg_index": [],
                    "weights": [],
                }
                continue
        
            idxs  = torch.tensor(per_class_selected_idx[c], device=self.device)
            ws    = torch.tensor(per_class_selected_w[c],  device=self.device)
        
            sorted_w, order = torch.sort(ws, descending=True)
            sorted_idxs = idxs[order]
            mask = sorted_w > w
        
            if mask.sum() == 0:   
                final_result[c] = {
                    "tg_index": [],
                    "weights": [],
                }
                continue
        
            filtered_idxs = sorted_idxs[mask]
            filtered_w    = sorted_w[mask]
            final_result[c] = {
                "tg_index": filtered_idxs,
                "weights":  filtered_w,
            }
        
        return final_result

    def compute_probability(self, feature3d):
        """
        Compute pseudo-label probabilities using nearest-anchor mapping.
    
        Args:
            prototypes: (Ca, d) prototype features
            feature3d: (B, d, H, W, D) 3D feature map
    
        Returns:
            probabilities: (B, C, H, W, D) probability map
            pseudo_labels: (B, H, W, D) pseudo-labels
        """
    
        device = self.device
        tau = 0.05  # temperature
    
        
        num_class = self.mu.shape[0]              # C
        B, d, H, W, D = feature3d.shape
        N = B * H * W * D                         
    
        d_volP_list = []
        for j in range(num_class):
            d_volP_j = torch.randn(self.bat_size_sr // num_class, self.dim, device=self.device) * self.sigma[j] + self.mu[j]
            d_volP_list.append(d_volP_j) 
        prototypes = torch.cat(d_volP_list, dim=0).to(device) # (Ca, d)
        Ca, d = prototypes.shape                  # Ca = C * a
    
        sr_class_id = torch.arange(num_class, device=device).repeat_interleave(
            self.bat_size_sr // num_class
        )  # (Ca,)
    
        anchor_ft = self.tg_fea.to(device)       # (b, d)
        b = anchor_ft.shape[0]
    
        # ¦Õ_j = h*_j + 1/2 ||q_j||^2
        d_temp_h = self.d_h.to(device)           # (b,)
        phi = d_temp_h + 0.5 * torch.sum(anchor_ft ** 2, dim=1)  # (b,)
    
        # ---------------- P(p_i = q_j) ----------------
        logits_p_q = torch.mm(prototypes, anchor_ft.t()) - phi.unsqueeze(0)  # (Ca, b)
        P_p_q = F.softmax(logits_p_q / tau, dim=1)                            # (Ca, b)
    
        # ---------------- feature flatten ----------------
        feature_flat = feature3d.permute(0, 2, 3, 4, 1).contiguous().view(N, d)  # (N, d)
        anchor_norm = F.normalize(anchor_ft, p=2, dim=1)  # (b, d)
        probabilities = torch.zeros(num_class, N, device=device)
        pseudo_labels = torch.zeros(N, device=device, dtype=torch.long)
        
        chunk_size = 31360 
        
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            f_chunk = feature_flat[start:end]  # (n, d)
            n = f_chunk.shape[0]
    
            f_norm = F.normalize(f_chunk, p=2, dim=1)        # (n, d)
            sim_matrix = torch.mm(f_norm, anchor_norm.t())   # (n, b)
            omega, max_sim_idx = sim_matrix.max(dim=1)  # (n,)
            
            phi_f = d_temp_h[max_sim_idx] + 0.5 * torch.sum(f_chunk ** 2, dim=1)  # (n,) 
            logits_p_f = (torch.mm(prototypes, f_chunk.t()) - phi_f.unsqueeze(0))                                           # (Ca, n)
            P_p_f = torch.softmax(logits_p_f / tau, dim=1)  # (Ca, n)

            tilde_s = (P_p_q[:, max_sim_idx] * omega.unsqueeze(0)
                      + P_p_f * (1.0 - omega).unsqueeze(0))  # (Ca, n)    
    
            # -------- s_r^c --------
            s_rc = torch.zeros(num_class, n, device=device)
            for c in range(num_class):
                mask = (sr_class_id == c)
                s_rc[c] = tilde_s[mask].sum(dim=0)
    
            # normalize over classes
            s_rc = s_rc / (tilde_s.sum(dim=0, keepdim=True) + 1e-8)
    
            # -------- pseudo labels --------
            prob, label = torch.max(s_rc, dim=0)
            probabilities[:, start:end] = s_rc
            pseudo_labels[start:end] = label
    
        probabilities = probabilities.view(num_class, B, H, W, D)
        pseudo_labels = pseudo_labels.view(B, H, W, D)
    
        return probabilities.permute(1, 0, 2, 3, 4), pseudo_labels


    def train_ot(self, target_measures, steps, optimization_num):
        best_g_norm = 1e20
        curr_best_g_norm = 1e20
        count_bad = 0
        h_file_list = []
        
        ot_dir= self.out_dir +"ot_la/" #args['h_name']
        if not os.path.exists(ot_dir):       
            os.makedirs(ot_dir)
        
        d_adam_m = torch.zeros(self.num_tg, dtype=torch.float, device=self.device)
        d_adam_v = torch.zeros(self.num_tg, dtype=torch.float, device=self.device)
        count_bad = 0
        count_double = 0
        while(steps <= self.max_iter):
            self.cal_measure()	
            
            bias_grd = self.d_g - target_measures
            d_adam_m *= 0.9
            d_adam_m += 0.1*bias_grd
            d_adam_v *= 0.999
            d_adam_v += 0.001*bias_grd*bias_grd
            d_delta_h = -self.lr*torch.div(d_adam_m, torch.add(torch.sqrt(d_adam_v),1e-8))
            self.d_h = self.d_h + d_delta_h
            #self.d_h = self.d_h - self.lr*bias_grd#It will cause the loss to decline very slowly
            '''normalize h'''
            self.d_h -= torch.mean(self.d_h)
            
            g_norm = torch.sqrt(torch.sum(torch.mul(bias_grd,bias_grd)))
              
            if (steps+1) % 50 == 0:    
                num_zero = torch.sum(self.d_g == 0)
                ratio_diff = torch.max(bias_grd)   
                '''print('[{0}/{1}] Max absolute error ratio: {2:.3f}. g norm: {3:.6f}. num zero: {4:d}'.format(
                    steps, self.max_iter, ratio_diff, g_norm, num_zero))'''
                 
            ''' /h: save  intercept vector of brenier_h function 
            '''
            if g_norm<curr_best_g_norm:
                filename = os.path.join(ot_dir,'h_best_{}.pt'.format(optimization_num))
                torch.save(self.d_h,filename)
                curr_best_g_norm = g_norm
                count_bad = 0
            else:
                count_bad += 1
                
            if steps+1 % 100 ==0 or steps+1 == self.max_iter:
                filename = os.path.join(ot_dir,'h_{}.pt'.format(optimization_num))
                torch.save(self.d_h,filename)
                #h_file_list.append(filename)
            
            if len(h_file_list)>6:
                if os.path.exists(h_file_list[0]):
                    os.remove(h_file_list[0])
                h_file_list.pop(0)
            
            if g_norm < 8e-4 and num_zero==0:
                return  
            
            if count_bad > 50 and count_double<3:
                self.num_bat_sr *= 2
                #print('self.num_bat_sr has increased to {}'.format(self.bat_size_sr*self.num_bat_sr))
                count_bad = 0
                curr_best_g_norm = 1e20
                self.lr *= 0.8
                count_double +=1
            
            steps += 1


    def set_h(self, h_tensor):
        self.d_h.copy_(h_tensor)     
