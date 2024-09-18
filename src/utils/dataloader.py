from torch.utils.data import Dataset
import torch
import numpy as np
import os
import cv2
import random
cv2.setNumThreads(1)


class BaseDataset(Dataset):
    def __init__(self, image_list_file):
        self.image_file_list = self.load_file_list(image_list_file)

    def load_file_list(self, filenames_file):
        image_file_list = []
        with open(filenames_file) as f:
            lines = f.readlines()
            for line in lines:
                image_file_list.append(line.strip().split())
        return image_file_list

    def load_image(self, path):
        img = cv2.imread(path)
        if img is None:
            print("not finding {}.".format(path))
            raise Exception("If the extension is different, set an argumentthe \"extension\" when you call dataloaders \"e.g. LoadFromImageFile\"")
        return img

    def resize_img(self, img, width, height):
        return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


class LoadFromImageFile(BaseDataset):
    def __init__(self, data_path, filenames_file, seed=None, train=True, transform=None, monocular=True, extension=None):
        super(LoadFromImageFile, self).__init__(filenames_file)
        np.random.seed(seed)
        random.seed(seed)
        self.root = data_path
        self.transform = transform
        self.monocular = monocular
        self.extension = extension
        if train:
            print('=> Load {} images from the pathes listed in {}'.format(len(self.image_file_list), self.root + "/" + filenames_file))

    def __getitem__(self, idx):
        if self.monocular:
            left_fn = self.image_file_list[idx][0]
            if self.extension:
                left_fn = os.path.splitext(left_fn)[0]
                image_path = os.path.join(self.root, left_fn) + self.extension
            else:
                image_path = os.path.join(self.root, left_fn)
            image = self.load_image(image_path)
            image = np.expand_dims(image, axis=0)
            if self.transform is not None:
                image = self.transform(image)
            sample = {"left": image[0]}
        else:
            left_fn, right_fn = self.image_file_list[idx]
            if self.extension:
                left_fn, right_fn = os.path.splitext(left_fn)[0], os.path.splitext(right_fn)[0]
                left_image_path = os.path.join(self.root, left_fn) + self.extension
                right_image_path = os.path.join(self.root, right_fn) + self.extension
            else:
                left_image_path = os.path.join(self.root, left_fn)
                right_image_path = os.path.join(self.root, right_fn)
            left_image = self.load_image(left_image_path)
            right_image = self.load_image(right_image_path)

            if self.transform is not None:
                images = [left_image, right_image]
                images = self.transform(images)
            sample = {"left": image[0], "right": images[1]}
        return sample

    def __len__(self):
        return len(self.image_file_list)


class SingleImageLoader(BaseDataset):
    def __init__(self, img_path, seed=None, transform=None):
        np.random.seed(seed)
        random.seed(seed)
        self.img_path = img_path
        self.transform = transform
        print('=> Load image from {}'.format(self.img_path))

    def __getitem__(self, idx):
        image = self.load_image(self.img_path)
        image = np.expand_dims(image, axis=0)
        if self.transform is not None:
            image = self.transform(image)
        sample = {"left": image[0]}
        return sample

    def __len__(self):
        return 1
