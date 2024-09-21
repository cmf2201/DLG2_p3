import torch
from PIL import Image
from torchvision.io import read_image
from torchvision.transforms.functional import perspective
from torchvision.transforms import ToPILImage
from torchvision.transforms import v2

to_image = ToPILImage()

img = read_image('/home/ctnguyen/neural_nemesis/DLG2_p3/render.jpg').to(torch.float64)
img = img / 255

transform = v2.RandomPerspective(distortion_scale=0.6, p=1.0)
transform1 = v2.RandomInvert(p=1.0)
transform2 = v2.RandomRotation(degrees=(0, 359))
transform3 = v2.GaussianNoise(mean=0.0, sigma=0.1, clip=True)

transform4 = transform1(transform)

# new_img = transform1(img)
new_img = transform3(img)
new_img = perspective(new_img, [(0,0),(1920,0),(1920,1080),(0,1080)], [(100,200),(1820,100),(1820,980),(100,980)])
new_img = perspective(new_img, [(100,200),(1820,100),(1820,980),(100,980)], [(0,0),(1920,0),(1920,1080),(0,1080)])
new_img = to_image(new_img)
new_img.convert('RGB')

new_img.save("/home/ctnguyen/neural_nemesis/DLG2_p3/here.png")