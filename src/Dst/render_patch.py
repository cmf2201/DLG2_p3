import numpy as np
from PIL import Image as im
import os
from torchvision.transforms import ToPILImage
import torch

to_image = ToPILImage()

directory_to_load = "/home/cmfrench/RBE474X/DLG2_p3/src/Dst/checkpoints/result"


def get_files_with_extension(folder_path, extension):
    """ Gets the path of all files with a given extension in a folder recursively """
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        # Only process directories that contain 'image_02'
        for file in files:
            if file.endswith(extension):
                file_paths.append(os.path.join(root, file))
    return file_paths

filepaths = get_files_with_extension(directory_to_load,"patch.npy")
for file in filepaths:
    array = np.load(file)
    tarray = torch.from_numpy(array)
    image = to_image(tarray)
    orig_name = (os.path.basename(file).split('/')[-1])
    orig_name = orig_name[0:-4]
    new_path = os.path.join("PatchCheckpoints",orig_name)

    image.save(f"{new_path}.png")