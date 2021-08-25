import json
import os
from typing import T_co, Text

import numpy as np
from PIL import Image

from dataset.dataset import Dataset
import pandas as pd


class AssessorDataset(Dataset):
    def __init__(self, path: Text, alpha=False, split: Text = 'train'):
        super().__init__(path=path, json_path=path, split=split)
        self.alpha = alpha

    def __getitem__(self, index) -> T_co:
        image = Image.open(self.images[index])
        image = image.convert('RGBA') if self.alpha else image.convert('RGB')
        image = self.transform(image)

        gt = self.gt[index]

        return image, gt

    def _load_images_and_gt(self):
        self.images = []
        img_files = open(os.path.join(self.path, 'image_locations.txt')).readlines()
        self.gt = np.load(os.path.join(self.path, 'ious.npy'), allow_pickle=True).astype(np.float32)
        #df = pd.read_csv(os.path.join(self.path, 'images.csv'), header=None, sep='\t')
        #img_files = df[0]
        #self.gt = np.array(df[1], dtype=np.float32)

        for file in img_files:
            self.images.append(os.path.join(self.path, file.replace('\n', '')))

        return self.images, self.gt
