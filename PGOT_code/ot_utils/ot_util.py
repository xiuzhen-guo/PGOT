import torch
import numpy as np
import scipy.io as sio
import os
from glob import glob
from ot_utils.optimal_transport import OptimalTransport
from einops import rearrange, repeat

torch.set_printoptions(precision=8)

#  generate latent code P
def transfer_and_generate(OT_solve, args, device):
    
    topk = args['topk']
    I_all = OT_solve.transfer_topk(topk)
    I_all = I_all.squeeze(0)
    
    if I_all.shape[0] != topk:
        I_all = I_all.permute(1, 0)  # [topk, numX]
    
    numX = I_all.shape[1]
    I_all_2 = -torch.ones([2, (topk-1) * numX], dtype=torch.long, device=device)
    
    for ii in range(topk-1):
        start, end = ii * numX, (ii+1) * numX
        I_all_2[0, start:end] = I_all[0, :].squeeze()  
        I_all_2[1, start:end] = I_all[ii+1, :].squeeze() 
    
    I_all = I_all_2
    
    if torch.sum(I_all < 0) > 0:
        print('Error: numX is not a multiple of bat_size_n !!!!!!!')

    #compute angles
    P = OT_solve.tg_fea.view(OT_solve.num_tg, -1) 
    nm = torch.cat([P, -torch.ones([OT_solve.num_tg,1],device=device)], dim=1)
    nm /= torch.norm(nm,dim=1).view(-1,1)
    cs = torch.sum(nm[I_all[0,:],:] * nm[I_all[1,:],:], 1) #element-wise multiplication
    cs = torch.min(torch.ones([cs.shape[0]],device=device), cs)
    theta = torch.acos(cs)
    theta = (theta-torch.min(theta))/(torch.max(theta)-torch.min(theta))

    #filter out TLerated samples with theta larger than threshold
    I_TL = I_all[:, theta <= args['angle_thresh']]
    I_TL, _ = torch.sort(I_TL, dim=0)
    _, uni_TL_id = np.unique(I_TL[0,:].cpu().numpy(), return_index=True)
    np.random.shuffle(uni_TL_id)
    I_TL = I_TL[:, torch.from_numpy(uni_TL_id)]
    
    #target features transfer   
    selected_indice = I_TL[0,:]  
    P_TL = OT_solve.tg_fea[I_TL[0,:]] 
    
    id_TL = I_TL[0,:].squeeze().cpu().numpy().astype(int)    
    TL_feature_path = os.path.join('./code/PG-OT/code/ot_utils','ot_target_features.mat')
    sio.savemat(TL_feature_path, {'features':P_TL.cpu().detach().numpy(), 'ids':id_TL})
    
    return P_TL, selected_indice
    

def topk_confidence(OT_solve, args, device):
    
    w = 0.5
    final_result = OT_solve.transfer_topk(w)    
    
    return final_result
    
def get_OT_solver(args, mu, sigma, tg_fea, optimization_num, self_snapshot_path):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    bat_size_tg = args['bat_size_tg']
    num_fea = tg_fea.shape[0]
    tg_measures = (torch.ones(num_fea)/num_fea).to(device)
    
    n_classes = mu.shape[0]   
    ot_solver = OptimalTransport(mu, sigma, tg_fea, args, device, self_snapshot_path)
    ot_solver.train_ot(tg_measures, 0, optimization_num) 

    return ot_solver
    
def OT_Map(args, mu, sigma, feature3d, optimization_num, self_snapshot_path):
    device = feature3d.device  
    sims = []
    # h
    ot_dir= self_snapshot_path +"ot_la/" 
    h_params = torch.load(f"{ot_dir}/h_best_{optimization_num}.pt", weights_only=True)
    # anchor_features
    anchor_data = torch.load(os.path.join(self_snapshot_path+"features/", "anchor_features.pt"))
    anchor_ft = anchor_data["anchor_features"]

    if mu is not []:
        solver = OptimalTransport(mu, sigma, anchor_ft, args, device, ot_dir)
        solver.set_h(h_params)
        probabilities, pseudo_labels = solver.compute_probability(feature3d)
        del solver  
    else:
        probabilities = torch.zeros_like(feature3d, dtype=torch.float32, device=device)
        pseudo_labels = torch.zeros_like(feature3d, dtype=torch.long, device=device)
        
    return probabilities, pseudo_labels
    
    
def transfer_and_remove_singular_points(args, mu, sigma, tg_fea, self_snapshot_path, optimization_num=1):
    device = tg_fea.device
    
    ot_dir = self_snapshot_path + "ot_la/" #args['h_name']
    if not os.path.exists(ot_dir):       
        os.makedirs(ot_dir)
    num_fea = tg_fea.shape[0]
    tg_measures = (torch.ones(num_fea)/num_fea).to(device)
    n_classes = mu.shape[0]   
    with torch.no_grad():
        ot_solver = OptimalTransport(mu, sigma, tg_fea, args, device, ot_dir)
        ot_solver.train_ot(tg_measures, 0, optimization_num=1)     
        h_params = torch.load(f"{ot_dir}/h_best_{optimization_num}.pt", weights_only=False)
        ot_solver.set_h(h_params)
        final_result = topk_confidence(ot_solver, args, device)
         
        del ot_solver
    
    torch.cuda.empty_cache()
    
    return final_result
    
def semi_discrete_OT_selecting_features(args, mu, sigma, tg_fea, unlab_ft_1d, self_snapshot_path, optimization_num=1):
    
    device = tg_fea.device
    ot_dir = self_snapshot_path + "ot_la/" 
    num_fea = tg_fea.shape[0]
    tg_measures = (torch.ones(num_fea)/num_fea).to(device)
    n_classes = mu.shape[0]   
    
    with torch.no_grad():
        ot_solver = OptimalTransport(mu, sigma, tg_fea, args, device, ot_dir)
        ot_solver.train_ot(tg_measures, 0, optimization_num=1)     
        h_params = torch.load(f"{ot_dir}/h_best_{optimization_num}.pt", weights_only=False)
        ot_solver.set_h(h_params)
        select_unlabeled_features, confidence_scores = OT_solve.transfer_reliable_features(unlab_ft_1d, w=0.5)
        topk_confidence(ot_solver, args, device)
         
        del ot_solver
    
    torch.cuda.empty_cache()
    if torch.cuda.memory_allocated() > 0.8 * torch.cuda.max_memory_allocated():
        torch.cuda.reset_peak_memory_stats()
    
    return final_result
    