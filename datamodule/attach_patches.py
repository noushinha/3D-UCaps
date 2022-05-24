import os.path
import time
import numpy as np
import nibabel as nib
import mrcfile as mrc
# define variables
num_class = 3
tomo_id = 23
overlap = 1
patch_size = 408
shape = (154, 409, 409)
shape2 = (154, 409, 409, 3)
# shape = (200, 512, 512)
# shape2 = (200, 512, 512, num_class)
# shape = (103, 409, 409)
# shape2 = (103, 409, 409, num_class)
bwidth = int(patch_size / 2)
patch_crop = 0
bcrop = int(bwidth - patch_crop)
slide = int(2 * bwidth + 1 - overlap)  # patch_size - overlap
tomo = np.zeros(shape).astype(np.float64)
avg_tomo = np.zeros(shape2).astype(np.int8)
label_map = np.zeros(shape).astype(np.int8)
binned_labelmap = np.zeros(shape2)

base_dir = '/mnt/Data/Cryo-ET/3D-UCaps/data/invitro/output/labelsTs'
# base_dir = '/mnt/Data/Cryo-ET/3D-UCaps/data/shrec/output/labelsTs'
# base_dir = '/mnt/Data/Cryo-ET/3D-UCaps/data/artificial/output/labelsTs'

def write_mrc(array, filename):
    """ This function writes an mrc file
        Args: filename: /saving/path
              array: nd array
    """
    mc = mrc.new_mmap(filename, shape=array.shape, mrc_mode=0, overwrite=True)
    for val in range(len(mc.data)):
        mc.data[val] = array[val]
    # read_mrc(filename)

# function definition
def correct_center_positions(xc, yc, zc, dim, offset):
    # If there are still few pixels at the end:
    if xc[-1] < dim[2] - offset:
        xc = xc + [dim[2] - offset, ]
    if yc[-1] < dim[1] - offset:
        yc = yc + [dim[1] - offset, ]
    # if zc[-1] < dim[0] - offset:
    #     zc = zc + [dim[0] - offset, ]

    return xc, yc, zc


x_centers = list(range(bwidth, tomo.shape[2] - bwidth, slide))
y_centers = list(range(bwidth, tomo.shape[1] - bwidth, slide))
z_centers = [100]  # 73 list(range(bwidth, tomo.shape[0] - bwidth, slide))

# if dimensions are not exactly divisible,
# we should collect the remained voxels around borders
x_centers, y_centers, z_centers = correct_center_positions(x_centers, y_centers, z_centers, tomo.shape, bwidth)

# total number of patches that we should extract
# just to check we have same number as we generated at the extraction time
total_pnum = len(x_centers) * len(y_centers) * len(z_centers)
start = time.time()

# two arrays to collect ther esults of rpediction
# one for predicted intensity values, second hodls the predicted class labels
pred_tvals = np.zeros(shape).astype(np.int8)
pred_tclass = np.zeros(shape + (num_class,)).astype(np.float16)  # tomo.shape * # classes
patch_num = 1
for z in z_centers:
    for y in y_centers:
        for x in x_centers:
            print('patch number ' + str(patch_num) + ' out of ' + str(total_pnum))
            # read nifti patches in order
            # put them in the huge tomogram
            # sum and average over the overlap regions
            patch_num_nifti = patch_num - 1
            patch_tomo_folder = "patch_m" + str(tomo_id) + "_p" + f'{patch_num_nifti:04d}'
            patch_tomo_name = patch_tomo_folder + "/patch_m" + str(tomo_id) + "_p" + f'{patch_num_nifti:04d}' \
                + "_ucaps_prediction.nii.gz"
            # patch_tomo_name = "patch_m08_p" + f'{patch_num_nifti:04d}' + ".nii.gz"
            patch_tomo_name = os.path.join(base_dir, patch_tomo_name)
            patch = nib.load(patch_tomo_name)
            patch = patch.get_fdata()
            print(np.unique(patch))
            # assign predicted values to the corresponding patch location in the tomogram
            # current_patch = tomo[z-bwidth:z+bwidth, y-bwidth:y+bwidth, x-bwidth:x+bwidth]
            # summed_values = patch + current_patch
            # tomo[z-bwidth:z+bwidth, y-bwidth:y+bwidth, x-bwidth:x+bwidth] = summed_values
            # if np.any(patch < 0):
            #     print("negative in patch")
            #
            # if np.any(current_patch < 0):
            #     print("negative in current_patch")
            #     exit()

            int_patch = patch.astype(np.int8)
            # current_patch2 = avg_tomo[z-bwidth:z+bwidth, y-bwidth:y+bwidth, x-bwidth:x+bwidth, :]
            current_patch2 = avg_tomo[:, y - bwidth:y + bwidth, x - bwidth:x + bwidth, :]
            xx, yy, zz = patch.shape

            # for i in range(0, 1):
            #     for j in range(0, yy):
            #         current_patch2[:, j, i, int_patch[:, j, i]] += 1
            for i in range(0, xx):
                for j in range(0, yy):
                    for k in range(0, zz):
                        if int_patch[i, j, k] == 0:
                            current_patch2[i, j, k, 0] += 1
                        elif int_patch[i, j, k] == 1:
                            current_patch2[i, j, k, 1] += 1
                        elif int_patch[i, j, k] == 2:
                            current_patch2[i, j, k, 2] += 1
                        elif int_patch[i, j, k] == 3:
                            current_patch2[i, j, k, 3] += 1
            # avg_tomo[z - bwidth:z + bwidth, y - bwidth:y + bwidth, x - bwidth:x + bwidth, :] = current_patch2
            avg_tomo[:, y - bwidth:y + bwidth, x - bwidth:x + bwidth, :] = current_patch2

            patch_num = patch_num + 1


# write_mrc(avg_tomo,  '/mnt/Data/Cryo-ET/DeepET/data2/results/RealData/evaluation/segment/scoremap_tomo.mrc')


# required only if there are overlapping regions (normalization)
label_map = np.argmax(avg_tomo, 3)

end = time.time()
processed_in = end - start
print("--- Processed in: ", processed_in, " ---")
print(np.unique(label_map))
# write_mrc(label_map,  '/mnt/Data/Cryo-ET/DeepET/data2/results/RealData/evaluation/segment/tomo_labelmap.mrc')
label_map = np.swapaxes(label_map, 0, 2)
empty_header_tomo = nib.Nifti1Header()
empty_header_tomo.get_data_shape()
nifti_mask = nib.Nifti1Image(np.array(label_map, dtype=np.uint8), np.eye(4))
patch_tomo_name = "mask_" + str(tomo_id) + ".nii.gz"
patch_tomo_name = os.path.join(base_dir, patch_tomo_name)
nib.save(nifti_mask, patch_tomo_name)
print(patch_tomo_name + " is saved")

# to use the result in amira perform flip x, flip y
