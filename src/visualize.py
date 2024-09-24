import torch
import argparse
import cv2
import numpy as np
import os
import utils.custom_transforms as transformer
from utils.patch_transformer import Adversarial_Patch
from models.adversarial_models import AdversarialModels
from utils.utils import makedirs, to_cuda_vars
from utils.dataloader import SingleImageLoader
from utils.visualizer import InputGradient, ActivationMap
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.simplefilter('ignore')

parser = argparse.ArgumentParser(description='Generating Adversarial Patches')
parser.add_argument('--img_path', type=str, default='Src/input_img/test_scene1.jpg')
parser.add_argument('--distill_ckpt', type=str, default="models/guo/distill_model.ckpt")
parser.add_argument('--height', type=int, help='input image height', default=256)
parser.add_argument('--width', type=int, help='input image width', default=512)
parser.add_argument('--seed', type=int, help='seed for random functions, and network initialization', default=0)
parser.add_argument('--patch_size', type=int, help='Resolution of patch', default=256)
parser.add_argument('--patch_shape', type=str, help='circle or square', default='circle')
parser.add_argument('--patch_path', type=str, default='Src/trained_patches/circle_patch_distill_near.npy')
parser.add_argument('--mask_path', type=str, default='Src/trained_patches/circle_mask.npy')
parser.add_argument('--model', nargs='*', type=str, default='distill', choices=['distill'], help='Model architecture')
parser.add_argument('--name', type=str, help='output directory', default='')
args = parser.parse_args()


def main():
    if args.name:
        save_path = os.path.join('Dst/visualization_result', args.name)
    else:
        name = (os.path.splitext(args.patch_path)[0]).replace('/', '_')
        save_path = os.path.join('Dst/visualization_result', name)
    print('===============================')
    print('=> Everything will be saved to \"{}\"'.format(save_path))
    makedirs(save_path)

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    # DataLoader
    test_transform = transformer.Compose([
        transformer.ResizeImage(h=args.height, w=args.width),
        transformer.ArrayToTensor()
    ])

    test_set = SingleImageLoader(
        img_path=args.img_path,
        seed=args.seed,
        transform=test_transform
    )

    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=1)

    print('===============================')
    # Attacked Models
    models = AdversarialModels(args)
    models.load_ckpt()

    # Patch and Mask
    adv = Adversarial_Patch(
        patch_type=args.patch_shape,
        batch_size=1,
        image_size=min(args.width, args.height),
        patch_size=args.patch_size,
        train=False
    )

    patch_cpu, mask_cpu = adv.load_patch_and_mask_from_file(args.patch_path, args.mask_path, npy=True)

    print('===============================')
    for i_batch, sample in enumerate(test_loader):
        sample = to_cuda_vars(sample)  # send item to gpu
        sample.update(models.get_original_disp(to_cuda_vars(sample)))  # get non-attacked disparity

        img, original_disp = sample['left'], sample['original_distill_disp']
        patch, mask = patch_cpu.cuda(), mask_cpu.cuda()

        # transform patch
        patch_t, mask_t = adv.patch_transformer(
            patch=patch,
            mask=mask,
            batch_size=1,
            img_size=min(args.width, args.height),
            do_rotate=True,
            rand_loc=True,
            random_size=True,
            train=False
        )

        # apply transformed patch to clean image
        attacked_img = adv.patch_applier(img, patch_t, mask_t)

        if 'distill' in args.model:
            ### visualize potentially_affected_regions ###
            print("Start calculating potentially affected regions ...")

            mask_nonzero = np.nonzero(mask_t[0, 0].data.cpu().numpy())
            heatmap = np.zeros((img.shape[2:]))

            for i in tqdm(range(args.height), desc='1st loop'):
                for j in tqdm(range(args.width), desc='2nd loop', leave=False):
                    cal_gradient = InputGradient(models.distill, args.height, args.width)
                    gradients = cal_gradient(img, i, j)
                    gradients = np.sum(gradients, axis=0)
                    heatmap[i, j] = np.abs(gradients[mask_nonzero]).sum()

            np.save(save_path+"/potentially_affected_regions.npy", heatmap)

            ### visualize activation map ###
            print("Start calculating activation map ...")

            mask_nonzero = np.nonzero(mask_t[0, 0].data.cpu().numpy())
            mask_coord = np.concatenate([mask_nonzero[0].reshape(-1, 1), mask_nonzero[1].reshape(-1, 1)], axis=1)

            original_activation_map = np.zeros((args.height, args.width), dtype=np.float32)
            attacked_activation_map = np.zeros((args.height, args.width), dtype=np.float32)

            cal_activation = ActivationMap(models.distill, "", args.height, args.width)

            for mask_h, mask_w in tqdm(mask_coord, total=len(mask_coord)):
                original_activation = cal_activation(img, mask_h, mask_w)
                attacked_activation = cal_activation(attacked_img, mask_h, mask_w)

                original_activation_map += original_activation
                attacked_activation_map += attacked_activation

            original_activation_map = ActivationMap.normalize_map(original_activation_map, args.height, args.width)
            attacked_activation_map = ActivationMap.normalize_map(attacked_activation_map, args.height, args.width)

            save_original_img = ActivationMap.show_maps_on_image(img[0].data.cpu().numpy(), original_activation_map)
            save_attacked_img = ActivationMap.show_maps_on_image(attacked_img[0].data.cpu().numpy(), attacked_activation_map)

            cv2.imwrite(save_path+"/activation_map_original_img.jpg", save_original_img)
            cv2.imwrite(save_path+"/activation_map_attacked_img.jpg", save_attacked_img)

    print("Finish test !")


if __name__ == '__main__':
    main()
