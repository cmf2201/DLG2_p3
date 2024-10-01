import torch
import torchvision
import argparse
import numpy as np
import time
import os
from models.adversarial_models import AdversarialModels
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


warnings.simplefilter('ignore')

parser = argparse.ArgumentParser(description='Generating Adversarial Patches')
parser.add_argument('--data_root', type=str, help='path to dataset', default='dataset')
parser.add_argument('--train_list', type=str, default='Src/list/eigen_train_list.txt')
parser.add_argument('--print_file', type=str, default='Src/list/printable30values.txt')
parser.add_argument('--distill_ckpt', type=str, default="repository/release-StereoUnsupFt-Mono-pt-CK.ckpt")
parser.add_argument('--height', type=int, help='input image height', default=256)
parser.add_argument('--width', type=int, help='input image width', default=512)
parser.add_argument('-b', '--batch_size', type=int, help='mini-batch size', default=60)
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

save_path = 'Dst/checkpoints/' + args.name
print('===============================')
print('=> Everything will be saved to \"{}\"'.format(save_path))
makedirs(save_path)

torch.manual_seed(args.seed)
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()


# setup your torchvision/anyother transforms here. This is for adding noise/perspective transforms and other changes to the background

# Transforms
def transforms(patch, mask):
    # Random size
    size_ran = (30,70)
    rand_size = random.randint(*size_ran)
    patch = v2.Resize(rand_size)(patch)
    mask = v2.Resize(rand_size)(mask)

    # Random Perspective
    patch2,mask2 = perspective_transformer(patch,mask)

    # Random Rotation
    rotate_range = (-15,15)
    rand_rotate = random.randint(*rotate_range)
    patch3 = v2.functional.rotate(patch2,rand_rotate)
    mask3 = v2.functional.rotate(mask2,rand_rotate)

    #Photometirc Distortional
    # patch = v2.RandomPhotometricDistort()(patch)

    return patch3,mask3

train_set = LoadFromImageFile(
    args.data_root,
    args.train_list,
    mask_path=args.mask_path,
    seed=args.seed,
    train=True,
    monocular=True,
    transform=None,
    extension=".png"
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_threads,
    pin_memory=True,
    drop_last=True
)

print('===============================')
torch.set_printoptions(profile="full")

# Patch and Mask
# Initialize a random patch image, resize to fix within training images
patch_cpu = torchvision.io.read_image(args.patch_path).float()/255
patch_cpu = patch_cpu.requires_grad_()
# print(patch_cpu)
mask_cpu = torchvision.io.read_image(args.mask_path).float()/255
# mask_cpu = v2.Resize(size=(56,56))(mask_cpu).requires_grad_()

# for i in range(10):
#     img, mask = transforms(patch_cpu,mask_cpu)
#     img = to_image(img)
#     mask = to_image(mask)

#     img.save(f"distortions/patch{i}.png")
#     mask.save(f"distortions/mask{i}.png")

    # Loss
loss_md = AdversarialLoss(args)

# Attacked Models
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cuda:2')
print(torch.cuda.is_available())
models = AdversarialModels(args)
models.load_ckpt()

# Train
print('===============================')
print("Start training ...")

start_time = time.time()


for i_batch, sample in tqdm(enumerate(train_loader), desc=f'Running epoch ', total=len(train_loader), leave=False):
    with torch.autograd.detect_anomaly():
        
        # send our patch, background, and mask to GPU
        sample = to_cuda_vars(sample)  # send item to gpu
        patch, mask = patch_cpu.cuda(), mask_cpu.cuda()

        # apply our transformations TODO:(do per batch, make a list)
        patchs = []
        masks = []
        for i in range(args.batch_size):
            p,m = transforms(patch,mask)
            patchs.append(p)
            masks.append(m)
        
        ## DEBUGGING CODE
        # for i,patch,mask in zip(range(args.batch_size),patchs,masks):
        #     patch = to_image(patch)
        #     mask = to_image(mask)
        #     patch.save(f"PatchCheckpoints/patch{i}.png")
        #     mask.save(f"PatchCheckpoints/mask{i}.png")

        patched_imgs, big_masks = image_paste(args.batch_size, sample['left'].float()/255, patchs, masks)
        sample.update({'patch':patched_imgs*255})

        # for i,patch_t,mask_t in zip(range(args.batch_size),torch.tensor_split(patched_imgs,args.batch_size,dim=0),torch.tensor_split(big_masks,args.batch_size,dim=0)):
        #     print(f"size:{patch_t.size()}")
        #     patch_t = patch_t.squeeze()
        #     mask_t = mask_t.squeeze()
        #     patch_t = to_image(patch_t)
        #     mask_t = to_image(mask_t)
            # patch_t.save(f"PatchCheckpoints/patch{i}.png")
            # mask_t.save(f"PatchCheckpoints/mask{i}.png")

        # Run image through Depth Map, as well as original image.
        sample.update(models.get_original_disp(sample))
        sample.update(models.get_disp_mask(sample))
        # 'original_disparity'
        # 'disparity'

        # Loss function on disparity

        # Take original image, and put target depth over the patch.
        Actual = sample['disparity']
        Original = sample['original_disparity']
        Target_list = []

        # Create the target disparity (original disp + distance )
        for batch in range(args.batch_size):
            Target_disp_pic = (big_masks[batch]) * (args.target_disp / 255)
            Sample_no_patch = (1-big_masks[batch]) * Original[batch]
            Target_list.append(Target_disp_pic + Sample_no_patch)
        Target = torch.stack(Target_list,dim=0)
    

        # for i,act,exp in zip(range(args.batch_size),torch.tensor_split(Actual,args.batch_size,dim=0),torch.tensor_split(Target,args.batch_size,dim=0)):
        #     loss = Target[i] - Actual[i]
        #     loss = to_image(loss)
        #     loss.save(f"PatchCheckpoints/loss{i}.png")
        #     act = act.squeeze()
        #     exp = exp.squeeze()
        #     exp = to_image(exp)
        #     act = to_image(act)
        #     act.save(f"PatchCheckpoints/act{i}.png")
        #     exp.save(f"PatchCheckpoints/exp{i}.png")
        # for batch in args.batch_size:
        loss = loss_md.forward(Actual,Target,patch)

        def to_heatmap(tensor,fname):
            pilten = to_image(tensor)
            image_array = np.array(pilten)
            plt.imshow(image_array, cmap='plasma', interpolation='nearest')
            plt.colorbar() # Add a colorbar to interpret values
            plt.savefig(fname)

        def to_difference(tensor,fname):
            pilten = to_image(tensor)
            image_array = np.array(pilten)
            plt.imshow(image_array, cmap='Reds', interpolation='nearest')
            plt.colorbar() # Add a colorbar to interpret values
            plt.savefig(fname)


        #original image + disparity
        orig_img = sample['left'][0]
        orig_disp = sample['original_disparity'][0]
        orig_disp_ten = orig_disp
        orig_img = to_image(orig_img)
        orig_img.save("PatchCheckpoints/original_image.png")
        to_heatmap(orig_disp,"PatchCheckpoints/digital_original_disparity.png")

        #attacked image + disparity
        atck_img = sample['patch'][0]
        atck_disp = sample['disparity'][0]
        atck_disp_ten = atck_disp
        atck_img = to_image(atck_img)
        atck_img.save("PatchCheckpoints/attacked_image.png")
        to_heatmap(atck_disp,"PatchCheckpoints/digital_attacked_disparity.png")

        difference_digital = torch.abs(orig_disp_ten - atck_disp_ten)
        to_difference(difference_digital, "PatchCheckpoints/digital_difference.png")

        torch.cuda.empty_cache()

