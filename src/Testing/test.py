import torch
import numpy as np
import random
from torchvision.transforms import ToPILImage

to_image = ToPILImage()

# for i in range(20):
#     patch = np.load('/home/ctnguyen/neural_nemesis/DLG2_p3/src/Dst/checkpoints/result/epoch_' + str(i) + '_mask.npy')

#     patch = torch.Tensor(patch)

#     image = to_image(patch/255)
#     image.save('/home/ctnguyen/neural_nemesis/DLG2_p3/src/Testing/generated/patch_' + str(i) + '.png')


# pasted_img = (1 - mask_t) * img1 + mask_t * patch_t

# batch_count = 2

# # B * 3C * H * W
# background_img_stack = torch.Tensor([
#         [
#                  [[255, 255, 255, 255, 255, 255, 255],
#                   [255, 255, 255, 255, 255, 255, 255],
#                   [255, 255, 255, 255, 255, 255, 255],
#                   [255, 255, 255, 255, 255, 255, 255],
#                   [255, 255, 255, 255, 255, 255, 255]],
#                  [[255, 255, 255, 255, 255, 255, 255],
#                   [255, 255, 255, 255, 255, 255, 255],
#                   [255, 255, 255, 255, 255, 255, 255],
#                   [255, 255, 255, 255, 255, 255, 255],
#                   [255, 255, 255, 255, 255, 255, 255]],
#                  [[255, 255, 255, 255, 255, 255, 255],
#                   [255, 255, 255, 255, 255, 255, 255],
#                   [255, 255, 255, 255, 255, 255, 255],
#                   [255, 255, 255, 255, 255, 255, 255],
#                   [255, 255, 255, 255, 255, 255, 255]]
#         ],
#         [
#                  [[500, 500, 500, 500, 500, 500, 500],
#                   [500, 500, 500, 500, 500, 500, 500],
#                   [500, 500, 500, 500, 500, 500, 500],
#                   [500, 500, 500, 500, 500, 500, 500],
#                   [500, 500, 500, 500, 500, 500, 500]],
#                  [[500, 500, 500, 500, 500, 500, 500],
#                   [500, 500, 500, 500, 500, 500, 500],
#                   [500, 500, 500, 500, 500, 500, 500],
#                   [500, 500, 500, 500, 500, 500, 500],
#                   [500, 500, 500, 500, 500, 500, 500]],
#                  [[500, 500, 500, 500, 500, 500, 500],
#                   [500, 500, 500, 500, 500, 500, 500],
#                   [500, 500, 500, 500, 500, 500, 500],
#                   [500, 500, 500, 500, 500, 500, 500],
#                   [500, 500, 500, 500, 500, 500, 500]],
#         ] 
#         ])

# # B * 3C * H * W
# patch_stack = torch.Tensor([
#         [
#                      [[10, 10, 10],
#                       [10, 10, 10],
#                       [10, 10, 10]],
#                      [[100, 100, 100],
#                       [100, 100, 100],
#                       [100, 100, 100]],
#                      [[200, 200, 200],
#                       [200, 200, 200],
#                       [200, 200, 200]]
#         ],
#         [
#                      [[910, 910, 910],
#                       [910, 910, 910],
#                       [910, 910, 910]],
#                      [[9100, 9100, 9100],
#                       [9100, 9100, 9100],
#                       [9100, 9100, 9100]],
#                      [[9200, 9200, 9200],
#                       [9200, 9200, 9200],
#                       [9200, 9200, 9200]]
#         ]             
#         ])

# output_stack = []
# for batch_i in range(batch_count):
#     background_img = background_img_stack[batch_i]
#     patch = patch_stack[batch_i]
    
#     # MAKE RANDOMIZER
#     x_pad = background_img.size(dim=2) - patch.size(dim=2)
#     y_pad = background_img.size(dim=1) - patch.size(dim=1)

#     print('xpad: ' + str(x_pad))
#     x_rand = random.randint(0, x_pad)
#     print('ypad: ' + str(y_pad))
#     y_rand = random.randint(0, y_pad)

#     pad_amount = (x_rand, x_pad - x_rand, y_rand, y_pad - y_rand) #left, right, up, down
#     print(pad_amount)
#     new_patch = torch.nn.functional.pad(patch, pad_amount, "constant", 0)

#     mask = torch.zeros(background_img.size())
#     mask = mask + new_patch
#     mask.clamp_(0, 1)

#     output = (1 - mask) * background_img + (mask * new_patch)
#     output_stack.append(output)

# output_stack = torch.stack(output_stack, dim=0)
# print(output_stack)

def image_paste(batch_count, background_img_stack, patch_stack):
    output_stack = []
    for batch_i in range(batch_count):
        background_img = background_img_stack[batch_i]
        patch = patch_stack[batch_i]
        
        # MAKE RANDOMIZER
        x_pad = background_img.size(dim=2) - patch.size(dim=2)
        y_pad = background_img.size(dim=1) - patch.size(dim=1)

        print('xpad: ' + str(x_pad))
        x_rand = random.randint(0, x_pad)
        print('ypad: ' + str(y_pad))
        y_rand = random.randint(0, y_pad)

        pad_amount = (x_rand, x_pad - x_rand, y_rand, y_pad - y_rand) #left, right, up, down
        print(pad_amount)
        new_patch = torch.nn.functional.pad(patch, pad_amount, "constant", 0)

        mask = torch.zeros(background_img.size())
        mask = mask + new_patch
        mask.clamp_(0, 1)

        output = (1 - mask) * background_img + (mask * new_patch)
        output_stack.append(output)

    output_stack = torch.stack(output_stack, dim=0)
    return output_stack




# # patch = background_img.zero_()
# pad_amount = (1,1,1,1) #left, right, up, down
# new_patch = torch.nn.functional.pad(patch, pad_amount, "constant", 0)

# # B * 3_C * H * W
# mask = torch.zeros(background_img.size())
# mask = mask + new_patch
# mask.clamp_(0, 1)

# # B * 3_C * H * W
# output = (1 - mask) * background_img + (mask * new_patch)
# print(new_patch)
# print(mask)
# print(output)