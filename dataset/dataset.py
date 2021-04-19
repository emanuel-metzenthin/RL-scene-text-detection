import os
from os.path import basename
from typing import Text

import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from torchvision.transforms import Compose, Resize, ToTensor, Normalize


class Dataset(Dataset):
    def __init__(self, path: Text, split: Text = 'train'):
        self.path = path
        self.split = split
        self._load_images_and_gt()

    def __getitem__(self, index) -> T_co:
        image = Image.open(self.images[index])
        image = image.convert('RGB')
        image = self.transform(image)

        gt = self.gt[index]

        return image, gt

    @staticmethod
    def transform(image):
        transforms = Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

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
