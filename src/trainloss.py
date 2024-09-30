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
parser.add_argument('--target_disp', type=int, default=200)
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
    
    # Transforms
    def transforms(patch, mask):
        # Random size
        size_ran = (30,70)
        rand_size = random.randint(*size_ran)
        patch = v2.Resize(rand_size)(patch)
        mask = v2.Resize(rand_size)(mask)

        # Random Perspective
        patch,mask = perspective_transformer(patch,mask)

        # Random Rotation
        rotate_range = (-15,15)
        rand_rotate = random.randint(*rotate_range)
        patch = v2.functional.rotate(patch,rand_rotate)
        mask = v2.functional.rotate(mask,rand_rotate)

        #Photometirc Distortional
        patch = v2.RandomPhotometricDistort()(patch)

        return patch,mask
    
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

    # Patch and Mask
    # Initialize a random patch image, resize to fix within training images
    patch_cpu = torchvision.io.read_image(args.patch_path)
    mask_cpu = torchvision.io.read_image(args.mask_path)
    # mask_cpu = v2.Resize(size=(56,56))(mask_cpu).requires_grad_()
    
    # for i in range(10):
    #     img, mask = transforms(patch_cpu,mask_cpu)
    #     img = to_image(img)
    #     mask = to_image(mask)

    #     img.save(f"distortions/patch{i}.png")
    #     mask.save(f"distortions/mask{i}.png")
    
    # Optimizer
    # pass the patch to the optimizer
    optimizer = torch.optim.Adam([patch_cpu], lr=args.lr, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50)


    # Attacked Models
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cuda:2')
    print(torch.cuda.is_available())
    models = AdversarialModels(args)
    models.load_ckpt()

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
                

                # apply our patch to image
                patched_imgs, big_masks = image_paste(args.batch_size, sample['left'], patchs)
                sample.update({'patch':patched_imgs})
                sample.update({'masks':big_masks})

                for i,patch_t,mask_t in zip(range(args.batch_size),torch.tensor_split(patched_imgs,args.batch_size,dim=0),torch.tensor_split(big_masks,args.batch_size,dim=0)):
                    print(f"size:{patch_t.size()}")
                    patch_t = patch_t.squeeze()
                    mask_t = mask_t.squeeze()
                    patch_t = to_image(patch_t)
                    mask_t = to_image(mask_t)
                    patch_t.save(f"PatchCheckpoints/patch{i}.png")
                    mask_t.save(f"PatchCheckpoints/mask{i}.png")

                # Run image through Depth Map, as well as original image.
                sample.update(models.get_original_disp(sample))
                sample.update(models.get_disp_mask(sample))
                # 'original_disparity'
                # 'disparity'

                # Loss function on disparity

                # Take original image, and put target depth over the patch.
                Actual = sample['disparity']
                Target = sample['original_disparity']
                # Create the target disparity (original disp + distance )
                for batch in range(args.batch_size):
                    print(f"size:{big_masks.size()}")
                    Target_disp_pic = (big_masks[batch]) * args.target_disp
                    Sample_no_patch = (1-big_masks[batch]) * Target[batch]
                    Target[batch] = Target_disp_pic + Sample_no_patch
                    # test2 = torch.ones(big_masks.size()) * 124
                    test = to_image(Target_disp_pic)
                    test.save(f"PatchCheckpoints/test1.png")
                    test2 = to_image(Sample_no_patch)
                    test2.save(f"PatchCheckpoints/test2.png")
                    Target[batch] = Target_disp_pic ## Here update the target batch with the target depth where the patch is

                for i,act,exp in zip(range(args.batch_size),torch.tensor_split(Actual,args.batch_size,dim=0),torch.tensor_split(Target,args.batch_size,dim=0)):
                    act = act.squeeze()
                    exp = exp.squeeze()
                    exp = to_image(exp)
                    act = to_image(act)
                    act.save(f"PatchCheckpoints/act{i}.png")
                    exp.save(f"PatchCheckpoints/exp{i}.png")
                # for batch in args.batch_size:
                break

                # target_depths = target_depths * target_depth * 
                # target_depths = torch.stack(list_of_masks).cuda().type(torch.float32).requires_grad_()
                # target_depths = target_depths / 255 * target_depth
                print(f"gradient:{patch_cpu.grad}")
                # Expected = target_depths
                break

                l1_loss = torch.nn.L1Loss()
                loss = l1_loss(Expected,Actual)
                


                # sample.update(models.get_original_disp(sample))  # get non-attacked disparity
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

                directory = f"Testing/generated/attacked/{str(i_batch)}"
                if not os.path.exists(directory):
                    os.makedirs(directory)

                for batch_index in range(img.size(dim=0)):
                    # retreive transformed patch information
                    patch_t, mask_t, endpoints = perspective_transformer(patch, mask)
                    list_of_endpoints.append(endpoints)
                    
                    # convert both images to PIL images
                    background = to_image(img[batch_index])
                    patch_t = to_image(patch_t)
                    mask_t = to_image(mask_t)

                    # coordinate of top left of patch_t
                    random_coordinate = (random.randint(0, background.width - patch_t.width), random.randint(0, background.height - patch_t.height))
                    list_of_relative_coords.append(random_coordinate)
                    
                    # apply transformed patch to clean image
                    background.convert("RGBA")
                    background.paste(patch_t,random_coordinate,mask_t.convert("L"))
                    # background.save(os.path.join(directory,f"{batch_index}.png"))

                    img_with_patch = pil_to_tensor(background)
                    batch_of_img_with_patch.append(img_with_patch)

                    ## list of untransformed masks
                    list_of_masks.append(torch.unsqueeze(mask[0], dim=0))

                tensor_of_img_with_patch = torch.stack(batch_of_img_with_patch)/255

                sample.update({'patch':tensor_of_img_with_patch.cuda()})
                sample.update(models.get_disp_mask(sample))
                
                # Now have 0-1 disparity.
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

                    # visual_patch_disp = to_image(disp_of_patch)
                    # visual_patch_disp.save(os.path.join(directory,f"after_mask{i}.png"))



                # tensor_of_patch_depth = torch.stack(list_of_disp_of_patch).requires_grad_().cuda()
                # target_depth = 10/255
                # target_depths = torch.stack(list_of_masks).cuda() / 255 * target_depth
                # target_depths.requires_grad_()

                # Loss

                # loss class calulates nps_loss and tv_loss
                # loss = loss_md.forward(tensor_of_patch_depth, target_depths)
                # disp_loss = loss_md.disp_loss

                # Used to display the loss of the given epoch.
                ep_disp_loss += Expected.detach().cpu().numpy()
                ep_loss += loss.detach().cpu().numpy()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                models.distill.zero_grad()

                patch_cpu.data.clamp_(0, 255)  # keep patch in image range

                del patch_t, loss, Actual, Expected, target_depths, disp_tensor, sample# nps_loss, tv_loss
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
