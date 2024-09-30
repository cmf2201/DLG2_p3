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
from PIL import Image
import matplotlib.pyplot as plt

warnings.simplefilter('ignore')

parser = argparse.ArgumentParser(description='Generating Adversarial Patches')
parser.add_argument('--data_root', type=str, help='path to dataset', default='dataset')
parser.add_argument('--train_list', type=str, default='Src/list/filter_eigen_train_list.txt')
parser.add_argument('--print_file', type=str, default='Src/list/printable30values.txt')
parser.add_argument('--distill_ckpt', type=str, default="repository/release-StereoUnsupFt-Mono-pt-CK.ckpt")
parser.add_argument('--height', type=int, help='input image height', default=256)
parser.add_argument('--width', type=int, help='input image width', default=512)
parser.add_argument('-b', '--batch_size', type=int, help='mini-batch size', default=16)
parser.add_argument('-j', '--num_threads', type=int, help='data loading workers', default=0)
parser.add_argument('--lr', type=float, help='initial learning rate', default=1e-4)
parser.add_argument('--num_epochs', type=int, help='number of total epochs', default=20)
parser.add_argument('--seed', type=int, help='seed for random functions, and network initialization', default=0)
parser.add_argument('--patch_size', type=int, help='Resolution of patch', default=56)
parser.add_argument('--patch_shape', type=str, help='circle or square', default='circle')
parser.add_argument('--patch_path', type=str, help='Initialize patch from file', default="/home/ctnguyen/neural_nemesis/DLG2_p3/src/baseline_patch-no_background.png")
parser.add_argument('--mask_path', type=str, help='Initialize mask from file', default="/home/ctnguyen/neural_nemesis/DLG2_p3/src/baseline_patch-no_background.png")
parser.add_argument('--colors_path', type=str, help='Directory of printable colors', default="/home/ctnguyen/neural_nemesis/DLG2_p3/src/Src/list/printable30values.txt")
parser.add_argument('--target_disp', type=int, default=120)
parser.add_argument('--model', nargs='*', type=str, default='distill', choices=['distill'], help='Model architecture')
parser.add_argument('--name', type=str, help='output directory', default="result")
args = parser.parse_args()

to_image = ToPILImage()

def main():
    save_path = 'Dst/checkpoints/' + args.name
    print('===============================')
    print('=> Everything will be saved to \"{}\"'.format(save_path))
    makedirs(save_path)

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()


    # setup your torchvision/anyother transforms here. This is for adding noise/perspective transforms and other changes to the background
    train_transform = None
    
    train_set = LoadFromImageFile(
        args.data_root,
        args.train_list,
        seed=args.seed,
        train=True,
        monocular=True,
        transform=train_transform,
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
    # Attacked Models
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cuda:2')
    print(torch.cuda.is_available())
    models = AdversarialModels(args)
    models.load_ckpt()

    # Loss
    loss_md = AdversarialLoss(args)

    # Patch and Mask
    # Initialize a random patch image, resize to fix within training images
    patch_cpu = torchvision.io.read_image(args.patch_path)
    patch_cpu = v2.Resize(size=(56,56))(patch_cpu)
    mask_cpu = torchvision.io.read_image(args.mask_path)
    mask_cpu = v2.Resize(size=(56,56))(mask_cpu)

    
    # Optimizer
    # pass the patch to the optimizer
    optimizer = torch.optim.Adam([patch_cpu], lr=args.lr, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50)

    # Train
    print('===============================')
    print("Start training ...")
    torch.set_printoptions(profile="full")

    start_time = time.time()

    for epoch in range(args.num_epochs):
        ep_nps_loss, ep_tv_loss, ep_loss, ep_disp_loss = 0, 0, 0, 0
        ep_time = time.time()

        for i_batch, sample in tqdm(enumerate(train_loader), desc=f'Running epoch {epoch}', total=len(train_loader), leave=False):
            with torch.autograd.detect_anomaly():
                #sample.permute(0, 3, 1, 2)
                sample = to_cuda_vars(sample)  # send item to gpu
                
                sample.update(models.get_original_disp(sample))  # get non-attacked disparity



                
                img, original_disp = sample['left'], sample['original_distill_disp']
                patch, mask = patch_cpu.cuda(), mask_cpu.cuda()

                patches = torch.stack([patch, patch, patch, patch, patch, patch, patch, patch, patch, patch, patch, patch, patch, patch, patch, patch])

                pasted_imgs, big_masks = image_paste(args.batch_size, img, patches)

                for i in range(args.batch_size):
                    pasted_img = pasted_imgs[i]
                    PILpasted = to_image(pasted_img)
                    PILpasted.save('/home/ctnguyen/neural_nemesis/DLG2_p3/src/Testing/pasted_patch_' + str(i) + '.png')

                # orig = original_disp[0]
                # orig = to_image(orig)

                # image_array = np.array(orig)
                # plt.imshow(image_array, cmap='hot', interpolation='nearest')
                # plt.colorbar() # Add a colorbar to interpret values
                # plt.savefig('Testing/generated/heatmap.png')

                # orig.save('Testing/generated/original_disp.png')
                # transform patch and maybe the mask corresponding to the transformed patch(binary iamge)
                batch_of_img_with_patch = []
                list_of_masks = []
                list_of_relative_coords = []
                list_of_endpoints = []
                for batch_index in range(img.size(dim=0)):
                    patch_t, mask_t, endpoints = perspective_transformer(patch, mask)
                    list_of_endpoints.append(endpoints)
                    
                    img1 = to_image(img[batch_index])
                    img2 = to_image(patch_t)
                    # coordinate of top left of patch_t
                    random_coordinate = (random.randint(0, img1.width - img2.width), random.randint(0, img1.height - img2.height))
                    list_of_relative_coords.append(random_coordinate)
                    
                    # apply transformed patch to clean image
                    
                    img1.paste(img2,random_coordinate)
                    # img1.save("Testing/generated/imageinimage" + str(batch_index) + ".png")
                    pasted_img = (1 - mask_t) * img1 + mask_t * patch_t
                    img_with_patch = pil_to_tensor(img1)
                    batch_of_img_with_patch.append(img_with_patch)
                    list_of_masks.append(torch.unsqueeze(mask[0], dim=0))

                tensor_of_img_with_patch = torch.stack(batch_of_img_with_patch) / 255
                sample.update({'patch':tensor_of_img_with_patch.cuda()})
                sample.update(models.get_disp_mask(sample))

                disp_with_mask = sample['distill_mask']

                list_of_disp_of_patch = []
                for i in range(disp_with_mask.size(0)):
                    patch_depth = disp_with_mask[i][0]
                    mask_transform = list_of_masks[i][0]
                    random_coord = list_of_relative_coords[i]
                    endpoint = list_of_endpoints[i]

                    # visual_patch_disp = to_image(patch_depth)
                    # visual_patch_disp.save('Testing/generated/before_crop' + str(i) + '.png')

                    # print(random_coord)
                    disp_of_patch = crop(img=patch_depth, top=random_coord[1], left=random_coord[0], 
                                            height=mask_transform.size(1), width=mask_transform.size(0))
                    
                    disp_of_patch = torch.unsqueeze(disp_of_patch, 0)
                    disp_of_patch = untransform(disp_of_patch, endpoint)
                    disp_of_patch = torch.squeeze(disp_of_patch)

                    # visual_patch_disp = to_image(disp_of_patch)
                    # visual_patch_disp.save('Testing/generated/before_mask' + str(i) + '.png')

                    disp_of_patch = disp_of_patch * (mask_transform / 255)
                    disp_of_patch = torch.unsqueeze(disp_of_patch, 0)
                    list_of_disp_of_patch.append(disp_of_patch)

                    visual_patch_disp = to_image(disp_of_patch)
                    visual_patch_disp.save('Testing/generated/after_mask' + str(i) + '.png')
                
                tensor_of_patch_depth = torch.stack(list_of_disp_of_patch).requires_grad_().cuda()
                target_depth = 10/255
                target_depths = torch.stack(list_of_masks).cuda() / 255 * target_depth
                target_depths.requires_grad_()

                # Loss
                # loss class calulates nps_loss and tv_loss
                loss = loss_md.forward(tensor_of_patch_depth, target_depths)
                # nps_loss = loss_md.nps_loss
                # tv_loss = loss_md.tv_loss
                disp_loss = loss_md.disp_loss

                # Used to display the loss of the given epoch.
                # ep_nps_loss += nps_loss.detach().cpu().numpy()
                # ep_tv_loss += tv_loss.detach().cpu().numpy()
                ep_disp_loss += disp_loss.detach().cpu().numpy()
                ep_loss += loss.detach().cpu().numpy()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                models.distill.zero_grad()

                patch_cpu.data.clamp_(0, 255)  # keep patch in image range

                del patch_t, loss, disp_loss, # nps_loss, tv_loss
                torch.cuda.empty_cache()

        ep_disp_loss = ep_disp_loss/len(train_loader)
        # ep_nps_loss = ep_nps_loss/len(train_loader)
        # ep_tv_loss = ep_tv_loss/len(train_loader)
        ep_loss = ep_loss/len(train_loader)
        scheduler.step(ep_loss)

        ep_time = time.time() - ep_time
        total_time = time.time() - start_time
        print('===============================')
        print(' FIN EPOCH: ', epoch)
        print('TOTAL TIME: ', format_time(int(total_time)))
        print('EPOCH TIME: ', format_time(int(ep_time)))
        print('EPOCH LOSS: ', ep_loss)
        print(' DISP LOSS: ', ep_disp_loss)
        # print('  NPS LOSS: ', ep_nps_loss)
        # print('   TV LOSS: ', ep_tv_loss)
        np.save(save_path + '/epoch_{}_patch.npy'.format(str(epoch)), patch_cpu.data.numpy())
        np.save(save_path + '/epoch_{}_mask.npy'.format(str(epoch)), mask_cpu.data.numpy())


if __name__ == '__main__':
    main()
