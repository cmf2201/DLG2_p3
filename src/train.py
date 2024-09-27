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
from utils.dataloader import LoadFromImageFile
from utils.utils import *
from tqdm import tqdm
import warnings
from PIL import Image
import matplotlib.pyplot as plt

warnings.simplefilter('ignore')

parser = argparse.ArgumentParser(description='Generating Adversarial Patches')
parser.add_argument('--data_root', type=str, help='path to dataset', default='dataset')
parser.add_argument('--train_list', type=str, default='Src/list/eigen_train_list.txt')
parser.add_argument('--print_file', type=str, default='Src/list/printable30values.txt')
parser.add_argument('--distill_ckpt', type=str, default="repository/release-StereoUnsupFt-Mono-pt-CK.ckpt")
parser.add_argument('--height', type=int, help='input image height', default=256)
parser.add_argument('--width', type=int, help='input image width', default=512)
parser.add_argument('-b', '--batch_size', type=int, help='mini-batch size', default=1)
parser.add_argument('-j', '--num_threads', type=int, help='data loading workers', default=0)
parser.add_argument('--lr', type=float, help='initial learning rate', default=1e-4)
parser.add_argument('--num_epochs', type=int, help='number of total epochs', default=40)
parser.add_argument('--seed', type=int, help='seed for random functions, and network initialization', default=0)
parser.add_argument('--patch_size', type=int, help='Resolution of patch', default=256)
parser.add_argument('--patch_shape', type=str, help='circle or square', default='circle')
parser.add_argument('--patch_path', type=str, help='Initialize patch from file', default="/home/ctnguyen/neural_nemesis/DLG2_p3/src/baseline_patch.png")
parser.add_argument('--mask_path', type=str, help='Initialize mask from file', default="/home/ctnguyen/neural_nemesis/DLG2_p3/src/baseline_patch.png")
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


    # setup your torchvision/anyother transforms here. This is for adding noise/perspective transforms and other changes to the patch
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
    # Initialize a random patch image
    patch_cpu = torchvision.io.read_image("/home/cmfrench/RBE474X/DLG2_p3/src/baseline_patch.png")
    mask_cpu = torchvision.io.read_image("/home/cmfrench/RBE474X/DLG2_p3/src/mask.png")

    
    # Optimizer
    # pass the patch to the optimizer
    optimizer = torch.optim.Adam([patch_cpu], lr=args.lr, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50)

    printable_colors = []
    printable_color_lines = open('Src/list/printable30values.txt').readlines()
    for line in printable_color_lines:
        string_color_array = line.strip()
        string_color_array = string_color_array.split(',')

        color_array = []
        for color_float in string_color_array:
            color_array.append(float(color_float))

        printable_colors.append(color_array)
    printable_colors = torch.Tensor(printable_colors)

    # Train
    print('===============================')
    print("Start training ...")
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

                orig = original_disp[0]
                orig = to_image(orig)

                image_array = np.array(orig)
                plt.imshow(image_array, cmap='hot', interpolation='nearest')
                plt.colorbar() # Add a colorbar to interpret values
                plt.savefig('heatmap.png')

                # orig.save('original_disp.png')
                # transform patch and maybe the mask corresponding to the transformed patch(binary iamge)
                patch_t, mask_t = patch, mask

                # apply transformed patch to clean image
                
                #img.paste(patch,(int(img.width/2),int(img.height/2)))
                
                # Loss
                # calculate the distance loss
                disp_loss = torch.zeros(mask_cpu.size(), requires_grad=True)
                # loss class calulates nps_loss and tv_loss
                loss = loss_md.forward(torch.zeros(mask_cpu.size(), requires_grad=True),disp_loss)
                nps_loss = loss_md.nps_loss
                tv_loss = loss_md.tv_loss

                # Used to display the loss of the given epoch.
                ep_nps_loss += nps_loss.detach().cpu().numpy()
                ep_tv_loss += tv_loss.detach().cpu().numpy()
                ep_disp_loss += disp_loss.detach().cpu().numpy()
                ep_loss += loss.detach().cpu().numpy()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                models.distill.zero_grad()

                patch_cpu.data.clamp_(0, 1)  # keep patch in image range

                del patch_t, loss, nps_loss, tv_loss, disp_loss
                torch.cuda.empty_cache()

        ep_disp_loss = ep_disp_loss/len(train_loader)
        ep_nps_loss = ep_nps_loss/len(train_loader)
        ep_tv_loss = ep_tv_loss/len(train_loader)
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
        print('  NPS LOSS: ', ep_nps_loss)
        print('   TV LOSS: ', ep_tv_loss)
        np.save(save_path + '/epoch_{}_patch.npy'.format(str(epoch)), patch_cpu.data.numpy())
        np.save(save_path + '/epoch_{}_mask.npy'.format(str(epoch)), mask_cpu.data.numpy())


if __name__ == '__main__':
    main()
