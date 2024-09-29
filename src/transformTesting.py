from utils.utils import *
import argparse
import torchvision
import os 
from PIL import Image

#include arguments for paths
parser = argparse.ArgumentParser(description='TransformTesting')
parser.add_argument('--patch_path', type=str, help='Initialize patch from file', default="/home/cmfrench/RBE474X/DLG2_p3/src/baseline_patch.png")
parser.add_argument('--mask_path', type=str, help='Initialize mask from file', default="/home/cmfrench/RBE474X/DLG2_p3/src/mask.png")
parser.add_argument('--transform_out', type=str, help='Directory for output of test images', default="/home/cmfrench/RBE474X/DLG2_p3/src/Testing/Transform")
args = parser.parse_args()

to_image = ToPILImage()

if not os.path.exists(args.transform_out):
    os.makedirs(args.transform_out)

## read in our images 
patch = torchvision.io.read_image(args.patch_path)
patch = v2.Resize(size=(56,56))(patch)
mask = torchvision.io.read_image(args.mask_path)
mask = v2.Resize(size=(56,56))(mask)

##original image
orig_img = to_image(patch)
orig_img.save(os.path.join(args.transform_out,f"orig.png"))

## apply transformation
patch_t, mask_t, endpoints = perspective_transformer(patch, mask)

## warped img
warp_img = to_image(patch_t)
warp_msk = to_image(mask_t)
warp_img.save(os.path.join(args.transform_out,f"warpimg.png"))
warp_msk.save(os.path.join(args.transform_out,f"warpmsk.png"))

result = Image.new("RGBA", warp_img.size)
result.paste(warp_img,(0,0),warp_msk.convert("L"))
result.save(os.path.join(args.transform_out,f"imgmsk.png"))


## unapply transformation
untransformed = untransform(patch_t, endpoints)

untrans_img = to_image(untransformed)
untrans_img.save(os.path.join(args.transform_out,f"untrans.png"))