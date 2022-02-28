import os
import sys
import numpy as np
import nibabel as nib
import mrcfile as mrc
from lxml import etree

base_dir = "/mnt/Data/Cryo-ET/DeepET/data2/0InvitroTargets/"
# base_dir = "/mnt/Data/Cryo-ET/DeepET/data2/Invitro_PTClass/"
# base_dir = "/mnt/Data/Cryo-ET/DeepET/data2/Invitro_RBClass/"
output_dir = "/mnt/Data/Cryo-ET/3D-UCaps/data/invitro/"

# # base_dir = "/mnt/Data/Cryo-ET/DeepET/data2/DeepET_Tomo_Masks_3Class/"
# base_dir = "/mnt/Data/Cryo-ET/DeepET/data2/DeepET_Tomo_Masks_1Class/"
# output_dir = "/mnt/Data/Cryo-ET/3D-UCaps/data/shrec3/"

patch_size = 64
patch_shift = 0
patches_tomos = []
patches_masks = []
# class_names_list = ["1"]  # proteasome class
# class_names_list = ["2"]  # ribosome class
class_names_list = ["1", "2"] # proteasome 1 and ribosome 2
# class_names_list = ["11", "12"]
# class_names_list = ["6", "11", "12"]

def read_mrc(filename):
    with mrc.open(filename, mode='r+', permissive=True) as mc:
        mc.update_header_from_data()
        mrc_tomo = mc.data
    return mrc_tomo


def read_xml2(filename):
    tree = etree.parse(filename)
    objl_xml = tree.getroot()

    obj_list = []
    for p in range(len(objl_xml)):
        lbl = objl_xml[p].get('class_label')
        if lbl in class_names_list:
                object_id = objl_xml[p].get('obj_id')
                tomo_idx = objl_xml[p].get('tomo_idx')
                # lbl = objl_xml[p].get('class_label')
                x = objl_xml[p].get('x')
                y = objl_xml[p].get('y')
                z = objl_xml[p].get('z')

                if object_id is not None:
                    object_id = int(object_id)
                else:
                    object_id = p
                if tomo_idx is not None:
                    tomo_idx = int(tomo_idx)
                add_obj(obj_list, tomo_idx=tomo_idx, obj_id=object_id, label=int(lbl), coord=(float(z), float(y), float(x)))
    return obj_list


def add_obj(obj_list, label, coord, obj_id=None, tomo_idx=None, c_size=None):
    obj = {
        'tomo_idx': tomo_idx,
        'obj_id': obj_id,
        'label': label,
        'x': coord[2],
        'y': coord[1],
        'z': coord[0],
        'c_size': c_size
    }

    obj_list.append(obj)
    return obj_list


def get_patch_position(tomodim, p_in, obj, shiftr):
    x = int(obj['x'])
    y = int(obj['y'])
    z = int(obj['z'])

    # Add random shift to coordinates:
    x = x + np.random.choice(range(-shiftr, shiftr + 1))
    y = y + np.random.choice(range(-shiftr, shiftr + 1))
    z = z + np.random.choice(range(-shiftr, shiftr + 1))

    # Shift position if passes the borders:
    if x < p_in:
        x = p_in
    if y < p_in:
        y = p_in
    if z < p_in:
        z = p_in

    if x > tomodim[2] - p_in:
        x = tomodim[2] - p_in
    if y > tomodim[1] - p_in:
        y = tomodim[1] - p_in
    if z > tomodim[0] - p_in:
        z = tomodim[0] - p_in

    return x, y, z


def int2str(n):
    strn = ""
    if 0 <= n < 10:
        strn = "000" + str(n)
    elif 10 <= n < 100:
        strn = "00" + str(n)
    elif 100 <= n < 1000:
        strn = "0" + str(n)
    else:
        strn = str(n)
    return str(strn)

radis = [10, 13]
list_tomoID = [8, 10, 21, 23]
for i in range(0, 4):
    tomo_id = "0" + str(i)
    # img_mrc = base_dir + '/reconstruction_model_' + str(tomo_id) + '.mrc'
    # msk_mrc = base_dir + 'target_reconstruction_model_' + str(tomo_id) + '.mrc'
    img_mrc = base_dir + '/' + str(list_tomoID[i]) + '_resampled.mrc'
    msk_mrc = base_dir + 'target_' + str(list_tomoID[i]) + '_resampled.mrc'
    tomo = read_mrc(img_mrc)
    mask = read_mrc(msk_mrc)
    # mask = mask[:, 0:409, 0:409]

    # check if the tomogram and its mask are of the same size
    if tomo.shape != mask.shape:
        print("the tomogram and the target must be of the same size. " +
              str(tomo.shape) + " is not equal to " + str(mask.shape) + ".")
        sys.exit()

    patches_tomos.append(tomo)
    patches_masks.append(mask)
#
list_annotations = read_xml2(os.path.join(base_dir, "object_list_train.xml"))

mid_dim = int(np.floor(patch_size / 2))
print(len(list_annotations))
cnt = 0
tomo_idx_prev = -1
for i in range(0, len(list_annotations)):  #
    # find the tomo
    tomo_idx = int(list_annotations[i]['tomo_idx'])

    if tomo_idx_prev != tomo_idx:
        cnt = 0
        # break

    # if tomo_idx != 0:
    #     break

    # read_tomo
    sample_tomo = patches_tomos[tomo_idx]
    sample_mask = patches_masks[tomo_idx]

    # correct positions
    x, y, z = get_patch_position(patches_tomos[tomo_idx].shape, mid_dim, list_annotations[i], patch_shift)

    # extract the patch from tomo and mask
    patch_tomo = sample_tomo[z - mid_dim:z + mid_dim, y - mid_dim:y + mid_dim, x - mid_dim:x + mid_dim]
    # patch_tomo = (patch_tomo - np.mean(patch_tomo)) / np.std(patch_tomo)
    patch_mask = sample_mask[z - mid_dim:z + mid_dim, y - mid_dim:y + mid_dim, x - mid_dim:x + mid_dim]

    # radi = radis[0]
    # if list_annotations[i]['label'] == 2:
    #     radi = radis[1]


    # cleaned_mask = np.zeros((64, 64, 64))
    # centered_particle = patch_mask[32-radi:32 + radi, 32-radi:32 + radi, 32-radi:32 + radi]
    # cleaned_mask[32 - radi:32 + radi, 32 - radi:32 + radi, 32 - radi:32 + radi] = centered_particle
    # cleaned_mask = cleaned_mask==list_annotations[i]['label']
    # patch_mask = cleaned_mask

    # save the extracted patches and their masks as nifti
    empty_header_tomo = nib.Nifti1Header()
    empty_header_tomo.get_data_shape()
    nifti_tomo = nib.Nifti1Image(patch_tomo, np.eye(4))

    # saving the tomo
    patch_tomo_name = "imagesTr/patch_t" + str(tomo_idx) + "_p" + int2str(cnt) + ".nii.gz"
    # if tomo_idx == 3:
    #     patch_tomo_name = "imagesTs/patch_t" + str(tomo_idx) + "_p" + int2str(cnt) + ".nii.gz"
    patch_tomo_name = os.path.join(output_dir, patch_tomo_name)
    nib.save(nifti_tomo, patch_tomo_name)
    print(patch_tomo_name + " is saved")

    empty_header_mask = nib.Nifti1Header()
    empty_header_mask.get_data_shape()
    nifti_mask = nib.Nifti1Image(np.array(patch_mask, dtype=np.uint8), np.eye(4))

    # saving the mask
    patch_mask_name = "labelsTr/patch_m" + str(tomo_idx) + "_c" + int2str(cnt) + ".nii.gz"
    # if tomo_idx == 3:
    #     patch_mask_name = "labelsTs/patch_m" + str(tomo_idx) + "_c" + int2str(cnt) + ".nii.gz"
    patch_mask_name = os.path.join(output_dir, patch_mask_name)
    nib.save(nifti_mask, patch_mask_name)
    print(patch_mask_name + " is saved")
    tomo_idx_prev = tomo_idx
    cnt = cnt + 1
