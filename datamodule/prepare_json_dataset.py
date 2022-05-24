import os
import re
import json
from glob import glob

# generating a json.dataset file similar to that of hippocampus dataset
# dataset = "shrec"
dataset = "invitro"
# dataset = "artificial"
base_dir = "/mnt/Data/Cryo-ET/3D-UCaps/data/" + str(dataset)
# mode = "train"
mode = "validate"

data = dict()
data = {"name": "invitro",
        "description": "molecular Structure Segmentation",
        "reference": " MPI",
        "licence": "CC-BY-SA 4.0",
        "release": "experimental data",
        "tensorImageSize": "3D",
        "modality": {"0": "tomogram"},
        "labels": {"0": "bg", "1": "ptr", "2": "rb"},
        # "labels": {"0": "bg", "1": "ptr"},
        # "labels": {"0": "bg", "1": "rb"},
        # "labels": {"0": "bg", "1": "3gl1"},
        # "labels": {"0": "bg", "1": "1bxn"},
        # "labels": {"0": "bg", "1": "4dhq"},
        "numTraining": 2370,
        "numTest": 4,  # tomo 11
        }

list_tr_tomos_IDs = glob(os.path.join(base_dir, "imagesTr/*.nii.gz"))
list_tr_masks_IDs = glob(os.path.join(base_dir, "labelsTr/*.nii.gz"))
list_ts_tomos_IDs = glob(os.path.join(base_dir, "imagesTs/*.nii.gz"))
list_ts_masks_IDs = glob(os.path.join(base_dir, "labelsTs/*.nii.gz"))

list_tr_tomos_IDs.sort(key=lambda f: int(re.sub('\D', '', f)))
list_tr_masks_IDs.sort(key=lambda r: int(re.sub('\D', '', r)))
list_ts_tomos_IDs.sort(key=lambda f: int(re.sub('\D', '', f)))
list_ts_masks_IDs.sort(key=lambda r: int(re.sub('\D', '', r)))

if mode == "train":
    training_item = []
    for t in range(len(list_tr_tomos_IDs)):
        tomo = list_tr_tomos_IDs[t]
        tomo = str.replace(tomo, "/mnt/Data/Cryo-ET/3D-UCaps/data/" + str(dataset), ".")
        mask = list_tr_masks_IDs[t]
        mask = str.replace(mask, "/mnt/Data/Cryo-ET/3D-UCaps/data/" + str(dataset), ".")

        strdict = {"image": tomo, "label": mask}
        training_item.append(strdict)
    data["training"] = training_item

    test_item = []
    for t in range(len(list_ts_tomos_IDs)):
        tomo = list_ts_tomos_IDs[t]
        tomo = str.replace(tomo, "/mnt/Data/Cryo-ET/3D-UCaps/data/" + str(dataset), ".")
        mask = list_ts_masks_IDs[t]
        mask = str.replace(mask, "/mnt/Data/Cryo-ET/3D-UCaps/data/" + str(dataset), ".")

        strdict = {"image": tomo, "label": mask}
        test_item.append(strdict)
    data["test"] = test_item
else:
    test_item = []
    for t in range(len(list_ts_tomos_IDs)):
        tomo = list_ts_tomos_IDs[t]
        tomo = str.replace(tomo, "/mnt/Data/Cryo-ET/3D-UCaps/data/" + str(dataset), ".")
        mask = list_ts_masks_IDs[t]
        mask = str.replace(mask, "/mnt/Data/Cryo-ET/3D-UCaps/data/" + str(dataset), ".")

        strdict = {"image": tomo, "label": mask}
        test_item.append(strdict)
    data["training"] = test_item

    training_item = []
    for t in range(len(list_tr_tomos_IDs)):
        tomo = list_tr_tomos_IDs[t]
        tomo = str.replace(tomo, "/mnt/Data/Cryo-ET/3D-UCaps/data/" + str(dataset), ".")
        mask = list_tr_masks_IDs[t]
        mask = str.replace(mask, "/mnt/Data/Cryo-ET/3D-UCaps/data/" + str(dataset), ".")

        strdict = {"image": tomo, "label": mask}
        training_item.append(strdict)
    data["test"] = training_item

print(data)
with open('/mnt/Data/Cryo-ET/3D-UCaps/data/' + str(dataset) + '/dataset.json', 'w') as outfile:
    json.dump(data, outfile, indent=2)
    outfile.write('\n')
