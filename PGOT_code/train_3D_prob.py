import argparse
import logging
import sys

import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from einops import rearrange
import wandb
import yaml
from dataloaders.dataset import *
from networks.unet_proto_3d import UNetProto
from utils import losses, ramps, test_3d_patch, util
from scheduler.my_lr_scheduler import PolyLR
import time

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='LA.yml', help='Path to the config file')
parser.add_argument('--root_path', type=str, default='./data/LASeg/', help='Name of Experiment')
parser.add_argument('--exp', type=str, default="LA2018", help='experiment_name')  
parser.add_argument('--model', type=str, default='vnet', help='model_name')

parser.add_argument('--max_iterations', type=int, default=15000, help='maximum epoch number to train')  
parser.add_argument("--pretrainIter", type=int, default=3000, help="maximum iteration to pretrain") # 3000 
parser.add_argument("--linearIter", type=int, default=200, help="maximum iteration to pretrain") # 200
parser.add_argument('--max_samples', type=int, default=80, help='maximum samples to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=1e-2, help='segmentation network learning rate')  # 1e-2 8e-3
parser.add_argument('--patch_size', type=list, default=(112, 112, 80), help='patch size of network input')
parser.add_argument('--seed', type=int, default=1337, help='random seed') 
parser.add_argument('--num_classes', type=int, default=2, help='output channel of network')
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
# label and unlabel
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')  
parser.add_argument('--labelnum', type=int, default=16, help='labeled data')  
# costs
parser.add_argument('--embed_dim', type=int, default=32, help='the dimension of the mu') 
parser.add_argument('--consistency', type=float, default=2.0, help='consistency') 
parser.add_argument('--consistency_rampup', type=float, default=120.0, help='consistency_rampup')
parser.add_argument("--proto_w", type=float, default=5e-3, help="the weight of proto loss") 
parser.add_argument('--proto_update_interval', type=int, default=100, help='the interval iterations for proto updating') # 200
parser.add_argument('--sdot_update_interval', type=int, default=100, help='the interval iterations for proto updating')

parser.add_argument("--dice_w", type=float, default=0.5, help="the weight of dice loss")
parser.add_argument("--ce_w", type=float, default=0.5, help="the weight of ce loss")
parser.add_argument('--momentum', type=bool, default=False, help='whether use momentum to update protos')
parser.add_argument('--device', type=str, default="0", help='the device')
args = parser.parse_args()

with open(os.path.join('./code/configs/', args.config), 'r') as f:
    Config = yaml.safe_load(f)

os.environ["CUDA_VISIBLE_DEVICES"] = args.device

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    with torch.no_grad():
        alpha = min(1 - 1 / (global_step + 1), alpha)
    
        ema_model.prototypes_mu.mul_(alpha).add_(1 - alpha, model.prototypes_mu)
        ema_model.prototypes_sigma.mul_(alpha).add_(1 - alpha, model.prototypes_sigma)
    
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

# loss function
CE = torch.nn.CrossEntropyLoss(weight=None)
dice_loss = losses.DiceLoss(args.num_classes, weights=None)

def mix_loss(output, label):

    loss_ce = CE(output, label)
    output_soft = F.softmax(output, dim=1)
    loss_dice = dice_loss(output_soft, label)
        
    return loss_ce, loss_dice

def unsup_mix_loss(output, label, weight=None):

    loss_ce = (F.cross_entropy(output, label, reduction='none') * weight).mean()
    output_soft = F.softmax(output, dim=1)
    loss_dice = dice_loss(output_soft, label, mask=weight)
    
    return loss_ce, loss_dice

def select_features(ft_3d, num_samples=10000):
    
    device = ft_3d.device
    B, C, H, W, D = ft_3d.shape

    total = B * H * W * D
    flat_idx = torch.randperm(total, device=device)[:num_samples]

    ft_1d = rearrange(ft_3d, "b c h w d -> (b h w d) c")
    selected_ft = ft_1d[flat_idx]

    mask = torch.zeros((B, H, W, D), device=device, dtype=torch.bool)
    mask = mask.view(-1)
    mask[flat_idx] = 1
    mask = mask.view(B, H, W, D)

    return selected_ft, mask


def train(args, snapshot_path):

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True  
        torch.backends.cudnn.deterministic = False 
        
    def create_model(pretrain=False, ema=False):
        net = UNetProto(
            config=args.config,
            backbone=args.model,
            inchannel=1,
            nclasses=args.num_classes,
            proto_mu=None,
            proto_sigma=None,
            embed_dim=args.embed_dim,
            momentum=args.momentum,
        )
        net = net.cuda()
        if ema:
            for param in net.parameters():
                param.detach_()
        return net


    s_model = create_model(pretrain=False, ema=False)
    t_model = create_model(pretrain=False, ema=True)

    
    optimizer = optim.SGD(s_model.backbone3d.parameters(), lr=args.base_lr,
                          momentum=0.9, weight_decay=0.0001, nesterov=True)
                          
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = LAHeart(base_dir=args.root_path,
                       split='train',
                       transform=transforms.Compose([
                           RandomRotFlip(),
                           RandomCrop(args.patch_size),
                           ToTensor(),
                       ]))
                       
    labeled_idxs = list(range(args.labelnum))
    unlabeled_idxs = list(range(args.labelnum, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size,
                                          args.batch_size - args.labeled_bs)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    s_model.train()
    t_model.train()
         
    iter_num = 0
    best_dice = 0.0
    max_epoch = args.max_iterations // len(trainloader) + 1
    iterator = tqdm(range(0, max_epoch), ncols=70)
    
    logging.info("Start self_training")
    logging.info("{} iterations per epoch".format(len(trainloader)))
    
    for epoch in iterator:
        for _, sampled_batch in enumerate(trainloader):
                
            iter_num = iter_num + 1
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            label_batch_l = label_batch[:args.labeled_bs]

            if iter_num <= args.linearIter:
                cls_seg_3d, _ = s_model.warm_up(volume_batch[:args.labeled_bs])
                loss_cls_ce_3d, loss_seg_dice_3d = mix_loss(cls_seg_3d, label_batch_l)
                loss = args.ce_w * loss_cls_ce_3d + args.dice_w * loss_seg_dice_3d
                logging.info('training linear cls only ... iteration %d : avg loss : %f' % (iter_num, loss.item()))
                              
            elif iter_num <= args.pretrainIter:
                outputs = s_model(x=volume_batch[:args.labeled_bs], label=label_batch_l)
                proto_mu = outputs["proto_mu"]
                proto_sigma = outputs["proto_sigma"]
                feature3d = outputs['feature3d']
                cls_seg_3d = outputs["cls_seg_3d"]
                
                # sdot optimization
                if iter_num == args.linearIter+1 or iter_num % args.sdot_update_interval==0:
                    select_ft, select_pos = select_features(feature3d)
                    proto_pseudo_list = []
                    proto_weights_list = []
                    sdot_h = s_model.sdot_map(Config['OT'], proto_mu, proto_sigma, select_ft, 1, self_snapshot_path)
                    
                proto_prob, proto_labels = compute_probability(Config['OT'], proto_mu, proto_sigma, feature3d)
                # update the prototypes
                if iter_num % args.proto_update_interval ==0:              
                    labeled_label = label_batch_l.long()
                    label_onehot = torch.nn.functional.one_hot(labeled_label, num_classes=args.num_classes)                    
                    mask_p = label_onehot.permute(4, 0, 1, 2, 3).contiguous().float()
                    confidence_p = torch.ones_like(mask_p, dtype=torch.float, device=mask_p.device)                    
                    proportion = s_model.prototype_update(feature3d, label_batch_l, args.num_classes, mask_p, confidence_p)  
                
                loss_cls_ce, loss_seg_dice = mix_loss(cls_seg_3d, label_batch_l)
                loss_cls_3d = args.ce_w * loss_cls_ce + args.dice_w * loss_seg_dice

                loss_proto_ce, loss_proto_dice = mix_loss(proto_prob, label_batch_l)
                loss_proto_3d = args.ce_w * loss_proto_ce + args.dice_w * loss_proto_dice
                loss_proto_l = args.proto_w * loss_proto_3d

                loss = loss_cls_3d + loss_proto_l
                logging.info('train both cls ... iteration %d : avg loss : %f loss_cls_3d : %f loss_proto_l: %f'
                             % (iter_num, loss.item(), loss_cls_3d.item(), loss_proto_l.item()))
                
            elif iter_num > args.pretrainIter:  

                with torch.no_grad():
                    u_output = t_model(x=volume_batch[args.labeled_bs:], mode = "test")
                    max_probs, max_idx = torch.max(u_output["cls_seg_3d"], dim=1)  

                # CutMix 
                mix_volume_ul, mix_label_ul = util.cut_mix(volume_batch[args.labeled_bs:], max_idx) 
                volume_batch = torch.cat((volume_batch[:args.labeled_bs],mix_volume_ul),dim=0)
                max_idx = mix_label_ul
                label_batch = torch.cat((label_batch_l, max_idx), dim=0)
                
                outputs = s_model(x=volume_batch)
                proto_mu = outputs["proto_mu"]
                proto_sigma = outputs["proto_sigma"]
                cls_seg_3d = outputs["cls_seg_3d"]
                feature3d = outputs['feature3d']

                # sdot optimization
                if iter_num % args.sdot_update_interval==0:
                    select_ft, select_pos = select_features(feature3d)
                    proto_pseudo_list = []
                    proto_weights_list = []
                    sdot_h = s_model.sdot_map(Config['OT'], proto_mu, proto_sigma, select_ft, 1, self_snapshot_path)                
                proto_prob, proto_labels = compute_probability(Config['OT'], proto_mu, proto_sigma, feature3d)
                
                # update the prototypes
                if iter_num % args.proto_update_interval ==0:
                    # labeled feature selection
                    labeled_label = label_batch_l.long()
                    label_onehot = torch.nn.functional.one_hot(labeled_label, num_classes=args.num_classes)                    
                    mask_p_l = label_onehot.permute(4, 0, 1, 2, 3).contiguous().float()
                    confidence_p_l = torch.ones_like(mask_p_l, dtype=torch.float, device=mask_p_l.device)
                    # unlabeled feature selection
                    mask_p_u, confidence_p_u = s_model.sr_feature_selection_fixed(feature3d[args.labeled_bs:], self_snapshot_path)
                    mask_p = torch.cat([mask_p_l, mask_p_u], dim=1)
                    confidence_p = torch.cat([confidence_p_l, confidence_p_u], dim=1)
                    proportion = s_model.prototype_update(feature3d, label_batch, args.num_classes, mask_p, confidence_p)
                
                # supervised loss
                loss_proto_ce, loss_proto_dice = mix_loss(proto_prob[:args.labeled_bs], label_batch_l)
                loss_proto_3d = args.ce_w * loss_proto_ce + args.dice_w * loss_proto_dice
                loss_proto_l = args.proto_w * loss_proto_3d
                loss_cls_ce, loss_seg_dice = mix_loss(cls_seg_3d[:args.labeled_bs], label_batch_l)
                loss_cls_3d = args.ce_w * loss_cls_ce + args.dice_w * loss_seg_dice
                loss_l = loss_cls_3d + loss_proto_l               
                
                # unsupervised loss
                weights, id_u = torch.max(proto_prob[args.labeled_bs:], dim=1)           
                loss_proto_ce, loss_proto_dice = unsup_mix_loss(cls_seg_3d[args.labeled_bs:], proto_labels[args.labeled_bs:].long(), weights)
                loss_proto_3d = args.ce_w * loss_proto_ce + args.dice_w * loss_proto_dice
                loss_cls_ce, loss_seg_dice = mix_loss(cls_seg_3d[args.labeled_bs:], max_idx)#, weight=mask)
                loss_cls_3d = args.ce_w * loss_cls_ce + args.dice_w * loss_seg_dice
                loss_proto_u = args.proto_w * loss_proto_3d
                loss_u = loss_cls_3d + loss_proto_u #+ loss_kl  #loss_ot_geom

                consistency_weight = get_current_consistency_weight((iter_num - args.pretrainIter) // 100)
                loss = loss_l + consistency_weight * loss_u                

                logging.info(
                    'iteration %d : loss : %f loss_sup : %f loss_ul : %f loss_proto_l:%f loss_proto_u:%f consistency_weight : %f'
                      % (iter_num, loss.item(), loss_l.item(), loss_u.item(), loss_proto_l.item(), loss_proto_u.item(), consistency_weight))
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                update_ema_variables(s_model, t_model, args.ema_decay, iter_num)
            
            if iter_num >= args.pretrainIter and iter_num % 200 == 0:
                s_model.eval()
                with torch.no_grad(): 
                    avg_metric = test_3d_patch.var_all_case_LA(s_model, num_classes=args.num_classes, 
                                                                patch_size=args.patch_size, stride_xy=18, stride_z=4)
                if avg_metric[0] > best_dice:
                    best_dice = round(avg_metric[0], 4)
                    save_mode_path = os.path.join(self_snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, best_dice))
                    save_best_path = os.path.join(self_snapshot_path, '{}_best_model.pth'.format(args.model))
                    save_s_protomu_path = os.path.join(self_snapshot_path, 's_proto_mu.pt'.format(args.model))
                    save_s_protosigma_path = os.path.join(self_snapshot_path, 's_proto_sigma.pt'.format(args.model))
                    save_t_protomu_path = os.path.join(self_snapshot_path, 't_proto_mu.pt'.format(args.model))
                    save_t_protosigma_path = os.path.join(self_snapshot_path, 't_proto_sigma.pt'.format(args.model))
                    torch.save(s_model.state_dict(), save_mode_path)
                    torch.save(s_model.state_dict(), save_best_path)
                    torch.save(s_model.prototypes_mu, save_s_protomu_path)
                    torch.save(s_model.prototypes_sigma, save_s_protosigma_path)
                    torch.save(t_model.prototypes_mu, save_t_protomu_path)
                    torch.save(t_model.prototypes_sigma, save_t_protosigma_path)
                    logging.info("save best model to {}".format(save_mode_path))

                logging.info('iteration %d : dice_score : %f jd: %f hd95 : %f asd : %f' % 
                            (iter_num, avg_metric[0], avg_metric[1], avg_metric[2], avg_metric[3]))
                s_model.train()

            if iter_num % 1000 == 0:
                save_mode_path = os.path.join(self_snapshot_path, 'iter_{}.pth'.format(iter_num))
                torch.save(s_model.state_dict(), save_mode_path)

            torch.cuda.empty_cache()
            if iter_num >= args.max_iterations:
                break
        if iter_num >= args.max_iterations:
            iterator.close()
            break


if __name__ == "__main__":
    import shutil
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    self_snapshot_path = "./model/{}_{}_labeled_{}(trilinear)/cons{}_consrampup{}_protow{}_seed{}/".format(args.exp, args.labelnum, args.model, args.consistency, args.consistency_rampup, args.proto_w, args.seed)
    for snapshot_path in [self_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
        if not os.path.exists(snapshot_path + '/code'):
            os.makedirs(snapshot_path + '/code')

        shutil.copyfile("./train_3D_prob.py", snapshot_path + "/code/train_3D_prob.py")
        shutil.copyfile("./networks/unet_proto_3d.py", snapshot_path + "/code/unet_proto_3d.py")

    logging.basicConfig(filename=self_snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, self_snapshot_path)

