import torch
import random
from PIL import Image
from torchvision.io import read_image
from torchvision.transforms.functional import perspective
from torchvision.transforms import ToPILImage
from torchvision.transforms import v2

to_image = ToPILImage()

img = read_image('/home/ctnguyen/neural_nemesis/DLG2_p3/baseline_patch.png').to(torch.float64)
img = img / 255

# new_img = transform1(img)
top_left = (0,0)
top_right = (224,0)
bottom_right = (224,224)
bottom_left = (0,224)
original = [top_left, top_right, bottom_right, bottom_left]
transform = [(top_left[0] + 2 + random.randint(0,20),top_left[1] + 2 + random.randint(0,20)),
             (top_right[0] - 2 - random.randint(0,20),top_right[1] + 2 + random.randint(0,20)),
             (bottom_right[0] - 2 - random.randint(0,20),bottom_left[1] - 2 - 20),
             (bottom_left[0] + 2 + random.randint(0,20),bottom_left[1] - 2 - random.randint(0,20))]
rand_rot = random.randint(0,359)

colorer = v2.ColorJitter(brightness=random.random()/2, contrast=random.random(), hue=random.random())
rotationer = v2.RandomRotation(degrees=(rand_rot, rand_rot))
un_rotationer = v2.RandomRotation(degrees=(360-rand_rot, 360-rand_rot))
gaussianer = v2.GaussianNoise(mean=0.0, sigma=0.1, clip=True)
new_img = colorer(img)
new_img = gaussianer(new_img)

new_img = perspective(new_img, original, transform)
new_img = rotationer(new_img)

new_new_img = new_img

new_img = to_image(new_img)
new_img.convert('RGB')
new_img.save("/home/ctnguyen/neural_nemesis/DLG2_p3/here1.png")

new_new_img = un_rotationer(new_new_img)
new_new_img = perspective(new_new_img, transform, original)

new_new_img = to_image(new_new_img)
new_new_img.convert('RGB')
new_new_img.save("/home/ctnguyen/neural_nemesis/DLG2_p3/here2.png")