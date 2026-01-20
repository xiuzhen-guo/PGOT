import h5py
import math
import nibabel as nib
import numpy as np
from medpy import metric
import torch
import torch.nn.functional as F
from tqdm import tqdm
from skimage.measure import label
from utils import util

def converToSlice(input):
    D = input.size(-1)
    input2d = input[..., 0]
    for i in range(1, D):
        input2dtmp = input[..., i]
        input2d = torch.cat((input2d, input2dtmp), dim=0)
    return input2d

def getLargestCC(segmentation):
    labels = label(segmentation)
    if labels.max() != 0:
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    else:
        largestCC = segmentation
    return largestCC
    
def calculate_metric_percase(gt, pred):
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        jaccard = metric.binary.jc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        asd = metric.binary.asd(pred, gt)
        return np.array([dice, jaccard, hd95, asd])
    else:
        return np.zeros(4)

def var_all_case_LA(model, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4,noise="None",std=0.2,blur_kernal_size=(5,5)):
   
    with open('./data/LASeg/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = ["./data/LASeg/2018LA_Seg_Training Set/" + item.replace('\n', '') + "/mri_norm2.h5" for item in image_list]
    loader = tqdm(image_list)
    total_metric = 0.0
    for image_path in loader:
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:] 
        label = h5f['label'][:]
        if noise == "None":
            pass
        elif noise == "blur":
            image = util.blur_mri_volume(image,kernel_size=blur_kernal_size)
        elif noise == "speckle":
            image = util.add_speckle_noise(image,std=std)
        prediction, score_map = test_single_case(model, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        if np.sum(prediction)==0:
            metric = 0,0,0,0
        else:
            metric = calculate_metric_percase(prediction, label[:])
        total_metric += np.asarray(metric)
    avg_metric = total_metric / len(image_list)
    print('average metric is {}'.format(avg_metric))
    return avg_metric

def var_all_case_pancreas(model, num_classes, patch_size=(96, 96, 96), stride_xy=16, stride_z=16,noise="None",std=0.2,blur_kernal_size=(5,5)):
   
    with open('./data/Pancreas/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = ["./data/Pancreas/pancreas_data/" + item.replace('\n', '') + ".h5" for item in image_list]
    loader = tqdm(image_list)
    total_metric = 0.0
    for image_path in loader:
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:] 
        label = h5f['label'][:]
        if noise == "None":
            pass
        elif noise == "blur":
            image = util.blur_mri_volume(image,kernel_size=blur_kernal_size)
        elif noise == "speckle":
            image = util.add_speckle_noise(image,std=std)
        prediction, score_map = test_single_case(model, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        if np.sum(prediction)==0:
            metric = 0,0,0,0
        else:
            metric = calculate_metric_percase(prediction, label[:])
        total_metric += np.asarray(metric)
    avg_metric = total_metric / len(image_list)
    print('average metric is {}'.format(avg_metric))
    return avg_metric


def test_all_case(model, image_list, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4, save_result=True, test_save_path=None, preproc_fn=None, metric_detail=0, nms=0):
    
    loader = tqdm(image_list) if not metric_detail else image_list
    total_metric = 0.0
    ith = 0
    for image_path in loader:
        name = image_path.split('/')[-2]
        print(name)
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        if preproc_fn is not None:
            image = preproc_fn(image)
        prediction, score_map = test_single_case(model, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        if nms:
            prediction = getLargestCC(prediction)
            
        if np.sum(prediction)==0:
            single_metric = (0,0,0,0)
        else:
            single_metric = calculate_metric_percase(prediction, label[:])
            
        if metric_detail:
            print('%02d,\t%.5f, %.5f, %.5f, %.5f' % (ith, single_metric[0], single_metric[1], single_metric[2], single_metric[3]))

        total_metric += np.asarray(single_metric)
        
        if save_result:
            nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), test_save_path +  "%02d_pred.nii.gz" % ith)
            nib.save(nib.Nifti1Image(image[:].astype(np.float32), np.eye(4)), test_save_path + "%02d_img.nii.gz" % ith)
            nib.save(nib.Nifti1Image(label[:].astype(np.float32), np.eye(4)), test_save_path + "%02d_gt.nii.gz" % ith)
        ith += 1

    avg_metric = total_metric / len(image_list)
    print('average metric is {}'.format(avg_metric))
    
    with open(test_save_path+'../performance.txt', 'w') as f:
        f.writelines('average metric is {} \n'.format(avg_metric))
    return avg_metric
    

def calculate_metric_percase(pred, gt):
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        jc = metric.binary.jc(pred, gt)
        hd = metric.binary.hd95(pred, gt)
        asd = metric.binary.asd(pred, gt)
        return np.array([dice, jc, hd, asd])
    else:
        return np.zeros(4)


def test_single_case_BraTS2024(model, image, stride_xy, stride_z, patch_size, num_classes=1):
    m, w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2,w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2,h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2,d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad,wr_pad),(hl_pad,hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
    m,ww,hh,dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image[0].shape).astype(np.float32)
    cnt = np.zeros(image[0].shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y,hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[:,xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(test_patch.cpu(),axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()
                with torch.no_grad():
                    outputs = model(test_patch, mode= "test") # d,2,h,w
                    if isinstance(outputs,tuple):
                        y1,_ = outputs
                    else:
                        y1 = outputs["cls_seg_3d"] # proto_seg
                    y = F.softmax(y1, dim=1)

                y = y.cpu().data.numpy()
                y = y[0,:,:,:,:]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    score_map = np.max(score_map, axis=0)
    score_map = score_map/np.expand_dims(cnt,axis=0)
    label_map = (score_map[:]>0.5).astype(int)
    if add_pad:
        label_map = label_map[:,wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        score_map = score_map[wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
    return np.squeeze(score_map, axis=0), label_map


def test_single_case(model, image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2,w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2,h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2,d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad,wr_pad),(hl_pad,hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww,hh,dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y,hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch,axis=0),axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    outputs = model(test_patch, mode= "test")
                    y1 = outputs["cls_seg_3d"] 
                    y = F.softmax(y1, dim=1)

                y = y.cpu().data.numpy()
                y = y[0,1,:,:,:]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    score_map = score_map/np.expand_dims(cnt,axis=0)
    label_map = (score_map[0]>0.5).astype(int)
    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        score_map = score_map[:,wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
    return label_map, score_map
