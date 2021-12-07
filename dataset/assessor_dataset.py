import json
import os
from typing import T_co, Text
import numpy as np
import torch
from PIL import Image
from dataset.dataset import Dataset
import pandas as pd


class AssessorDataset(Dataset):
    def __init__(self, path: Text, alpha=False, dual_image=False, split: Text = 'train'):
        self.alpha = alpha
        self.dual_image = dual_image
        super().__init__(path=path, json_path=None, split=split)

    def __getitem__(self, index) -> T_co:
        image = Image.open(self.images[index])
        if self.dual_image:
            image = image.convert('RGBA') if self.alpha else image.convert('L')
            sur_image = Image.open(self.surrounding_images[index])
            sur_image = sur_image.convert("RGBA")
            image, sur_image = self.transform([image, sur_image])
            image, sur_image = image.unsqueeze(0), sur_image.unsqueeze(0)

            image = torch.vstack((image, sur_image))
        else:
            image = image.convert('RGBA') if self.alpha else image.convert('RGB')
            image = self.transform(image)
        gt = self.gt[index]

        return image, gt

    def _load_images_and_gt(self):
        self.images = []
        self.surrounding_images = []
        img_files = open(os.path.join(self.path, 'image_locations.txt')).readlines()

        img_files = [i.split(",") for i in img_files]

        self.gt = np.load(os.path.join(self.path, 'ious.npy'), allow_pickle=True).astype(np.float32)

        for file in img_files:
            if self.dual_image:
                img, sur_img = file
                self.images.append(os.path.join(self.path, img.replace('\n', '')))
                self.surrounding_images.append(os.path.join(self.path, sur_img.replace('\n', '')))
            else:
                self.images.append(os.path.join(self.path, file[0].replace('\n', '').replace('./', '')))

        return self.images, self.surrounding_images, self.gt
