import os
import numpy as np
import mrcfile as mrc
import nibabel as nib

# base_dir = "/mnt/Data/Cryo-ET/DeepET/data2/DeepET_Tomo_Masks_3Class"
base_dir = "/mnt/Data/Cryo-ET/DeepET/data2/Invitro/Invitro_PTRBClass"
# base_dir = "/mnt/Data/Cryo-ET/DeepET/data2/Invitro/Invitro_PTClass"
# base_dir = "/mnt/Data/Cryo-ET/DeepET/data2/Invitro/Invitro_RBClass"
# output_dir = "/mnt/Data/Cryo-ET/3D-UCaps/data/invitro/"

# base_dir = "/mnt/Data/Cryo-ET/DeepET/data2/SHREC/SHREC_3GL1"
# base_dir = "/mnt/Data/Cryo-ET/DeepET/data2/SHREC/SHREC_1BXN"
# base_dir = "/mnt/Data/Cryo-ET/DeepET/data2/SHREC/SHREC_4D8Q"
# base_dir = "/mnt/Data/Cryo-ET/DeepET/data2/SHREC/SHREC_MultiClass"
# output_dir = "/mnt/Data/Cryo-ET/3D-UCaps/data/shrec/"

# base_dir = "/mnt/Data/Cryo-ET/DeepET/data2/Artificial/MultiClass/"
output_dir = "/mnt/Data/Cryo-ET/3D-UCaps/data/artificial/"

overlap = 1
patch_num = 1
patch_size = 408  # 512
tomo_id = 23
bwidth = int(patch_size / 2)
slide = int(2 * bwidth + 1 - overlap)  # 1  # patch_size - overlap
dataset_type = "real"  # real, shrec, artificial

def read_mrc(filename):
    with mrc.open(filename, mode='r+', permissive=True) as mc:
        mc.update_header_from_data()
        mrc_tomo = mc.data
    return mrc_tomo


def correct_center_positions(xc, yc, zc, dim, offset):
    # If there are still few pixels at the end:
    if xc[-1] < dim[2] - offset:
        xc = xc + [dim[2] - offset, ]
    if yc[-1] < dim[1] - offset:
        yc = yc + [dim[1] - offset, ]
    # if zc[-1] < dim[0] - offset:
    #     zc = zc + [dim[0] - offset, ]

    return xc, yc, zc

if dataset_type == "real":
    tomo_name = os.path.join(base_dir, str(tomo_id) + '_resampled.mrc')
    mask_name = os.path.join(base_dir, 'target_' + str(tomo_id) + '_resampled.mrc')
elif dataset_type == "shrec":
    tomo_name = os.path.join(base_dir, 'reconstruction_model_0' + str(tomo_id) + '.mrc')
    mask_name = os.path.join(base_dir, 'target_reconstruction_model_0' + str(tomo_id) + '.mrc')
elif dataset_type =="artificial":
    tomo_name = os.path.join(base_dir, 'tomo_rec_' + str(tomo_id) + '.mrc')
    mask_name = os.path.join(base_dir, 'target_tomo_rec_' + str(tomo_id) + '.mrc')

tomo = read_mrc(tomo_name)
mask = read_mrc(mask_name)


x_centers = list(range(bwidth, tomo.shape[2] - bwidth, slide))
y_centers = list(range(bwidth, tomo.shape[1] - bwidth, slide))
z_centers = [77]  # [73], [100], [51]  # list(range(bwidth, tomo.shape[0] - bwidth, slide))

# if dimensions are not exactly divisible, we should collect the remained voxels around borders
x_centers, y_centers, z_centers = correct_center_positions(x_centers, y_centers, z_centers, tomo.shape, bwidth)

# total number of patches that we should extract
total_pnum = len(x_centers) * len(y_centers) * len(z_centers)
print("total # patches: ", total_pnum)
for z in z_centers:
    for y in y_centers:
        for x in x_centers:
            # patch = tomo[z - bwidth:z + bwidth, y - bwidth:y + bwidth, x - bwidth:x + bwidth]
            # patch_mask = mask[z - bwidth:z + bwidth, y - bwidth:y + bwidth, x - bwidth:x + bwidth]

            patch = tomo[:, y - bwidth:y + bwidth, x - bwidth:x + bwidth]
            patch_mask = mask[:, y - bwidth:y + bwidth, x - bwidth:x + bwidth]

            empty_header_tomo = nib.Nifti1Header()
            empty_header_tomo.get_data_shape()
            nifti_tomo = nib.Nifti1Image(patch, np.eye(4))
            patch_num_nifti = patch_num - 1
            patch_tomo_name = "imagesTs/patch_t" + str(tomo_id) + "_p" + f'{patch_num_nifti:04d}' + ".nii.gz"
            patch_tomo_name = os.path.join(output_dir, patch_tomo_name)
            nib.save(nifti_tomo, patch_tomo_name)
            print(patch_tomo_name + " is saved")

            nifti_patch = patch_mask
            empty_header_tomo = nib.Nifti1Header()
            empty_header_tomo.get_data_shape()
            nifti_mask = nib.Nifti1Image(np.array(patch_mask, dtype=np.uint8), np.eye(4))
            patch_num_nifti = patch_num - 1
            patch_tomo_name = "labelsTs/patch_m" + str(tomo_id) + "_p" + f'{patch_num_nifti:04d}' + ".nii.gz"
            patch_tomo_name = os.path.join(output_dir, patch_tomo_name)
            nib.save(nifti_mask, patch_tomo_name)
            print(patch_tomo_name + " is saved\n\n")

            patch_num = patch_num + 1
