import os.path
import numpy as np
import nibabel as nib

# patch = nib.load("/mnt/Data/Cryo-ET/DeepET/data/invitro_RibosomeAndProteasome/tomo_23/23_resampled.nii")
patch = nib.load("/mnt/Data/Cryo-ET/DeepET/data2/Invitro_RBClass/target_8_resampled.nii")
empty_header_tomo = nib.Nifti1Header()
empty_header_tomo.get_data_shape()
nifti_mask = nib.Nifti1Image(np.array(patch, dtype=np.uint8), np.eye(4))
patch_tomo_name = "mask_8.nii.gz"
patch_tomo_name = os.path.join(patch_tomo_name)
nib.save(nifti_mask, patch_tomo_name)
print(patch_tomo_name + " is saved")