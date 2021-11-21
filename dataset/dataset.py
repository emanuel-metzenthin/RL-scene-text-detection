import os
from os.path import basename
from typing import Text

import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from torchvision.transforms import Compose, Resize, ToTensor, RandomHorizontalFlip, RandomApply, Grayscale, Normalize, GaussianBlur, ColorJitter


class Dataset(Dataset):
    class AddGaussianNoise(object):
        def __init__(self, mean=0., std=1.):
            self.std = std
            self.mean = mean

        def __call__(self, tensor):
            return tensor + torch.randn(tensor.size()) * self.std + self.mean

        def __repr__(self):
            return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

    def __init__(self, path: Text, json_path: Text, split: Text = 'train', img_size=(224, 224)):
        self.path = path
        if json_path is None:
            self.json_path = path
        else:
            self.json_path = json_path
        self.split = split
        self.img_size = img_size
        self._load_images_and_gt()

    def __getitem__(self, index) -> T_co:
        image = Image.open(self.images[index])
        image = image.convert('RGB')
        image = self.transform(image)

        gt = self.gt[index]

        return image, gt

    def transform(self, image):
        augm = Compose([
            GaussianBlur(5),
            RandomApply([self.AddGaussianNoise(0, 0.1)], p=0.5),
            # ColorJitter(hue=0.2, saturation=0.2, contrast=0.25),
            RandomHorizontalFlip(),
        ])

        resize = Compose([
            Resize(self.img_size),
            ToTensor(),
        ])

        transforms = resize

        if type(image) == list:
            return [transforms(img) for img in image]

        return transforms(image)

    def __len__(self):
        return len(self.images)

    def _load_images_and_gt(self):
        raise NotImplementedError

    @staticmethod
    def collate_fn(batch):
        images = list()
        boxes = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])

        images = torch.stack(images, dim=0)

        return images, boxes
