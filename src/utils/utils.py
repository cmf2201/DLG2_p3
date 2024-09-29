import os
import time
import random
import torch
from torchvision.transforms.functional import perspective
from torchvision.transforms import ToPILImage
from torchvision.transforms import Pad
from torchvision.transforms import v2
from torchvision.io import read_image


def makedirs(save_path):
    os.makedirs(save_path, exist_ok=True)


def to_cuda_vars(vars_dict):
    new_dict = {}
    for k, v in vars_dict.items():
        if isinstance(v, torch.Tensor):
            new_dict[k] = v.float().cuda() / 255
    return new_dict


def format_time(time):
    hour, minute, second = time // 3600, (time % 3600) // 60, time % 3600 % 60
    string = str(hour).zfill(2) + ":" + str(minute).zfill(2) + ":" + str(second).zfill(2)
    return string


def nps_loss(img, colors):
    min_distance = 100
    nps = 0
    for i in range(img.shape[0]):
        for k in range (img.shape[1]):
            rgb = (img[i][k])
            rgb = rgb.unsqueeze(0)
            min_distance = 100
            for j in range(colors.shape[0]):
                distance = torch.cdist(rgb, colors[j])
                distance = torch.sum(distance)
                if(distance < min_distance):
                    min_distance = distance
            nps += min_distance
    return nps


def perspective_transformer(patch, mask):
    size = patch.size()[1:3] # 56, 56

    startpoints = ((0,0),
                   (0,size[1]),
                   (size[0],size[1]),
                   (size[0],0))
    
    endpoints = ((int(random.randint(0, size[0]) / 4), int(random.randint(0, size[0]) / 4)),
                 (int(random.randint(0, size[0]) / 4), size[1] - int(random.randint(0, size[0]) / 4)),
                 (size[0] - int(random.randint(0, size[0]) / 4), size[1] - int(random.randint(0, size[0]) / 4)),
                 (size[0] - int(random.randint(0, size[0]) / 4), int(random.randint(0, size[0]) / 4)))

    perspective_patch = perspective(img=patch, startpoints=startpoints,endpoints=endpoints)
    perspective_mask = perspective(img=mask, startpoints=startpoints,endpoints=endpoints)

    return (perspective_patch, perspective_mask, endpoints)

def untransform(img, endpoints):
    size = img.size()[1:3] # 56, 56

    startpoints = ((0,0),
                   (0,size[1]),
                   (size[0],size[1]),
                   (size[0],0))
    
    untransformed_mask = perspective(img=img, startpoints=endpoints, endpoints=startpoints)
    return untransformed_mask