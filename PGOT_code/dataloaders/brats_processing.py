import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage import measure
import nibabel as nib
import SimpleITK as sitk
import glob
import os

def brain_bbox(data, gt):
    mask = (data != 0)
    brain_voxels = np.where(mask != 0)
    minZidx = int(np.min(brain_voxels[0]))
    maxZidx = int(np.max(brain_voxels[0]))
    minXidx = int(np.min(brain_voxels[1]))
    maxXidx = int(np.max(brain_voxels[1]))
    minYidx = int(np.min(brain_voxels[2]))
    maxYidx = int(np.max(brain_voxels[2]))
    data_bboxed = data[minZidx:maxZidx, minXidx:maxXidx, minYidx:maxYidx]
    gt_bboxed = gt[minZidx:maxZidx, minXidx:maxXidx, minYidx:maxYidx]
    return data_bboxed, gt_bboxed


def volume_bounding_box(data, gt, expend=0, status="train"):
    data, gt = brain_bbox(data, gt)
    print(data.shape)
    mask = (gt != 0)
    brain_voxels = np.where(mask != 0)
    z, x, y = data.shape
    minZidx = int(np.min(brain_voxels[0]))
    maxZidx = int(np.max(brain_voxels[0]))
    minXidx = int(np.min(brain_voxels[1]))
    maxXidx = int(np.max(brain_voxels[1]))
    minYidx = int(np.min(brain_voxels[2]))
    maxYidx = int(np.max(brain_voxels[2]))

    minZidx_jitterd = max(minZidx - expend, 0)
    maxZidx_jitterd = min(maxZidx + expend, z)
    minXidx_jitterd = max(minXidx - expend, 0)
    maxXidx_jitterd = min(maxXidx + expend, x)
    minYidx_jitterd = max(minYidx - expend, 0)
    maxYidx_jitterd = min(maxYidx + expend, y)

    data_bboxed = data[minZidx_jitterd:maxZidx_jitterd,
                       minXidx_jitterd:maxXidx_jitterd, minYidx_jitterd:maxYidx_jitterd]
    print([minZidx, maxZidx, minXidx, maxXidx, minYidx, maxYidx])
    print([minZidx_jitterd, maxZidx_jitterd,
           minXidx_jitterd, maxXidx_jitterd, minYidx_jitterd, maxYidx_jitterd])

    if status == "train":
        gt_bboxed = np.zeros_like(data_bboxed, dtype=np.uint8)
        gt_bboxed[expend:maxZidx_jitterd-expend, expend:maxXidx_jitterd -
                  expend, expend:maxYidx_jitterd - expend] = 1
        return data_bboxed, gt_bboxed

    if status == "test":
        gt_bboxed = gt[minZidx_jitterd:maxZidx_jitterd,
                       minXidx_jitterd:maxXidx_jitterd, minYidx_jitterd:maxYidx_jitterd]
        return data_bboxed, gt_bboxed


def itensity_normalize_one_volume(volume):
    """
    normalize the itensity of an nd volume based on the mean and std of nonzeor region
    inputs:
        volume: the input nd volume
    outputs:
        out: the normalized nd volume
    """

    pixels = volume[volume > 0]
    mean = pixels.mean()
    std = pixels.std()
    out = (volume - mean)/std
    out_random = np.random.normal(0, 1, size=volume.shape)
#     out[volume == 0] = out_random[volume == 0]
    out = out.astype(np.float32)
    return out


class MedicalImageDeal(object):
    def __init__(self, img, percent=1):
        self.img = img
        self.percent = percent

    @property
    def valid_img(self):
        from skimage import exposure
        cdf = exposure.cumulative_distribution(self.img)
        watershed = cdf[1][cdf[0] >= self.percent][0]
        return np.clip(self.img, self.img.min(), watershed)

    @property
    def norm_img(self):
        return (self.img - self.img.min()) / (self.img.max() - self.img.min())


'''all_flair = glob.glob("/home/shiji/gxz/data/BraTS2024/validation_data/")
for p in all_flair:
    for q in p:   
        data = sitk.GetArrayFromImage(sitk.ReadImage(q))
        lab = sitk.GetArrayFromImage(sitk.ReadImage(q.replace("t2f", "seg")))
        img, lab = brain_bbox(data, lab)
        img = MedicalImageDeal(img, percent=0.999).valid_img
        img = itensity_normalize_one_volume(img)
        #lab[lab > 0] = 1
        uid = p.split("/")[-1]
        seg_uid = uid.replace("t2f","seg")
        print(uid)
        print(seg_uid)
        sitk.WriteImage(sitk.GetImageFromArray(
            img), "/home/amsshi/GXZ/data/BraTS2024/Data/t2f/{}".format(uid))'''

# Define paths
input_root = "/home/shiji/gxz/data/BraTS2024/training_data1_v2/"
output_root = "/home/shiji/gxz/data/BraTS2024/validation_data_preprocessed/"

print(f"Checking input directory: {input_root}")
print(f"Directory exists: {os.path.exists(input_root)}")
print(f"Directory contents: {os.listdir(input_root)}")

# Get all patient folders
patient_folders = glob.glob(os.path.join(input_root, "BraTS-GL-*"))
print(f"Found {len(patient_folders)} patient folders")

if not patient_folders:
    print("No patient folders found! Check your glob pattern and directory structure.")
    print("Trying alternative pattern...")
    patient_folders = glob.glob(os.path.join(input_root, "*"))
    print(f"Found {len(patient_folders)} items with alternative pattern")
    print("Items found:", patient_folders)

# Define all modalities to process
modalities = ['t1c', 't1n', 't2f', 't2w']

for patient_folder in patient_folders:
    patient_id = os.path.basename(patient_folder)
    print(f"\nProcessing patient: {patient_id}")
    print(f"Patient folder: {patient_folder}")
    
    output_patient_folder = os.path.join(output_root, patient_id)
    os.makedirs(output_patient_folder, exist_ok=True)
    print(f"Created output folder: {output_patient_folder}")
    
    # First process the segmentation if it exists
    seg_file = os.path.join(patient_folder, f"{patient_id}-seg.nii.gz")
    print(f"Looking for segmentation file: {seg_file}")
    print(f"Segmentation exists: {os.path.exists(seg_file)}")
    
    if os.path.exists(seg_file):
        try:
            print("Processing segmentation...")
            lab_data = sitk.GetArrayFromImage(sitk.ReadImage(seg_file))
            # Save original segmentation (we'll crop it when processing images)
            sitk.WriteImage(sitk.GetImageFromArray(lab_data), 
                           os.path.join(output_patient_folder, f"{patient_id}-seg.nii.gz"))
            print("Segmentation saved successfully")
        except Exception as e:
            print(f"Error processing segmentation for {patient_id}: {str(e)}")
    
    # Process each modality independently
    for modality in modalities:
        input_file = os.path.join(patient_folder, f"{patient_id}-{modality}.nii.gz")
        print(f"\nProcessing modality: {modality}")
        print(f"Looking for file: {input_file}")
        print(f"File exists: {os.path.exists(input_file)}")
        
        if not os.path.exists(input_file):
            print(f"File not found: {input_file}")
            continue
            
        try:
            # Read image data
            print("Reading image...")
            img_data = sitk.GetArrayFromImage(sitk.ReadImage(input_file))
            
            # If segmentation exists, process both image and segmentation together
            if os.path.exists(seg_file):
                print("Processing with segmentation...")
                lab_data = sitk.GetArrayFromImage(sitk.ReadImage(seg_file))
                img_data, lab_data = brain_bbox(img_data, lab_data)
                
                # Update the segmentation file (only needs to be done once, but harmless if repeated)
                sitk.WriteImage(sitk.GetImageFromArray(lab_data), 
                               os.path.join(output_patient_folder, f"{patient_id}-seg.nii.gz"))
                print("Updated segmentation with bounding box")
            
            # Process the image
            print("Applying MedicalImageDeal...")
            img_data = MedicalImageDeal(img_data, percent=0.999).valid_img
            print("Normalizing intensity...")
            img_data = itensity_normalize_one_volume(img_data)
            
            # Save processed image
            output_file = os.path.join(output_patient_folder, f"{patient_id}-{modality}.nii.gz")
            print(f"Saving to: {output_file}")
            sitk.WriteImage(sitk.GetImageFromArray(img_data), output_file)
            
            print(f"Successfully processed: {output_file}")
            
        except Exception as e:
            print(f"Error processing {input_file}: {str(e)}")
