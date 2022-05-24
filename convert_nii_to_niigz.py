import os.path
import numpy as np
import nibabel as nib


def read_mrc(filename):
    """ This function reads an mrc file and returns the 3D array
        Args: filename: path to mrc file
        Returns: 3d array
    """

    import mrcfile as mrc
    with mrc.open(filename, mode='r+', permissive=True) as mc:
        mc.update_header_from_data()
        mrc_tomo = mc.data
    return mrc_tomo


# patch = nib.load("/mnt/Data/Cryo-ET/data/Invitro/tomo_8/proteasome8AvgNoWei_p2.2A.mrc")
patch = read_mrc("/mnt/Data/Cryo-ET/data/Invitro/tomo_8/proteasome8AvgNoWei_p2.2A.resampled.mrc")
# patch = nib.load("/mnt/Data/Cryo-ET/3D-UCaps/data/invitro/labelsTs/target_23_resampled.mrc")
empty_header_tomo = nib.Nifti1Header()
empty_header_tomo.get_data_shape()
empty_header_tomo = nib.Nifti1Header()
nifti_mask = nib.Nifti1Image(np.array(np.swapaxes(patch, 0, 2), dtype=np.float), np.eye(4))
patch_tomo_name = "/mnt/Data/Cryo-ET/data/Invitro/tomo_8/proteasome8Avg.nii.gz"
patch_tomo_name = os.path.join(patch_tomo_name)
nib.save(nifti_mask, patch_tomo_name)
print(patch_tomo_name + " is saved")
