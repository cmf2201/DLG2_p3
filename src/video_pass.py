import torch
import torchvision
import argparse
import numpy as np
import time
import os
from models.adversarial_models import AdversarialModels
from torchvision.io import read_image
from models.loss_model import AdversarialLoss
from torchvision.transforms import v2
from torchvision.transforms import ToPILImage
from torchvision.transforms.functional import pil_to_tensor
from torchvision.transforms.functional import crop
from utils.dataloader import LoadFromImageFile
from utils.utils import *
from tqdm import tqdm
import warnings
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Generating Adversarial Patches')
parser.add_argument('--data_root', type=str, help='path to dataset', default='dataset')
parser.add_argument('--train_list', type=str, default='Src/list/eigen_train_list.txt')
parser.add_argument('--print_file', type=str, default='Src/list/printable30values.txt')
parser.add_argument('--distill_ckpt', type=str, default="repository/release-StereoUnsupFt-Mono-pt-CK.ckpt")
parser.add_argument('--height', type=int, help='input image height', default=256)
parser.add_argument('--width', type=int, help='input image width', default=512)
parser.add_argument('-b', '--batch_size', type=int, help='mini-batch size', default=4)
parser.add_argument('-j', '--num_threads', type=int, help='data loading workers', default=0)
parser.add_argument('--lr', type=float, help='initial learning rate', default=1e-4)
parser.add_argument('--num_epochs', type=int, help='number of total epochs', default=20)
parser.add_argument('--seed', type=int, help='seed for random functions, and network initialization', default=0)
parser.add_argument('--patch_size', type=int, help='Resolution of patch', default=56)
parser.add_argument('--patch_shape', type=str, help='circle or square', default='circle')
parser.add_argument('--patch_path', type=str, help='Initialize patch from file', default="/home/ctnguyen/neural_nemesis/DLG2_p3/src/baseline_patch.png")
parser.add_argument('--mask_path', type=str, help='Initialize mask from file', default="/home/ctnguyen/neural_nemesis/DLG2_p3/src/baseline_patch.png")
parser.add_argument('--colors_path', type=str, help='Directory of printable colors', default="/home/cmfrench/RBE474X/DLG2_p3/src/Src/list/printable30values.txt")
parser.add_argument('--target_disp', type=int, default=70)
parser.add_argument('--model', nargs='*', type=str, default='distill', choices=['distill'], help='Model architecture')
parser.add_argument('--name', type=str, help='output directory', default="result")
args = parser.parse_args()

to_image = ToPILImage()

def to_heatmap(tensor,fname):
            pilten = to_image(tensor)
            image_array = np.array(pilten)
            plt.imshow(image_array, cmap='plasma', interpolation='nearest')
            plt.rcParams['axes.spines.left'] = False
            plt.rcParams['axes.spines.right'] = False
            plt.rcParams['axes.spines.top'] = False
            plt.rcParams['axes.spines.bottom'] = False
            plt.savefig(fname)
            plt.clf()


before_folder = '/home/cmfrench/RBE474X/DLG2_p3/src/Before'
after_folder = '/home/cmfrench/RBE474X/DLG2_p3/src/After'

before_files = os.listdir(before_folder)
after_files = os.listdir(after_folder)

models = AdversarialModels(args)
models.load_ckpt()

all_before_images = []
for image_name in before_files:
    image = read_image(before_folder + '/' + image_name)
    all_before_images.append(image)

all_after_images = []
for image_name in after_files:
    image = read_image(after_folder + '/' + image_name)
    all_after_images.append(image)

before_stack = torch.stack(all_before_images)
after_stack = torch.stack(all_after_images)

for image_count in range(before_stack.size(dim=0)):
    single_image = torch.unsqueeze(before_stack[image_count], dim=0).cuda()
    single_image = v2.Resize(size=(256,512))(single_image)
    sample = {'left':single_image.float() / 255}
    original_disp_dict = models.get_original_disp(sample)
    disp_tensor = original_disp_dict["original_disparity"][0]
    image_count_string = str(image_count)
    to_heatmap(disp_tensor, '/home/cmfrench/RBE474X/DLG2_p3/src/Before_disp/' + (4 - len(image_count_string)) * '0' + image_count_string + '.png')
    del single_image, sample, original_disp_dict, disp_tensor

for image_count in range(after_stack.size(dim=0)):
    single_image = torch.unsqueeze(after_stack[image_count], dim=0).cuda()
    single_image = v2.Resize(size=(256,512))(single_image)
    sample = {'left':single_image.float() / 255}
    original_disp_dict = models.get_original_disp(sample)
    disp_tensor = original_disp_dict["original_disparity"][0]
    image_count_string = str(image_count)
    to_heatmap(disp_tensor, '/home/cmfrench/RBE474X/DLG2_p3/src/After_disp/' + (4 - len(image_count_string)) * '0' + image_count_string + '.png')
    del single_image, sample, original_disp_dict, disp_tensor