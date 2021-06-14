import json
import os
import numpy as np
from dataset.dataset import Dataset
import pandas as pd


class AssessorDataset(Dataset):
    def _load_images_and_gt(self):
        self.images = []
        img_files = open(os.path.join(self.path, 'image_locations.txt')).readlines()
        self.gt = np.load(os.path.join(self.path, 'ious.npy'), allow_pickle=True)
        # df = pd.read_csv(os.path.join(self.path, 'images.csv'), header=None, sep='\t')
        # img_files = df[0]
        # self.gt = df[1]

        for file in img_files:
            self.images.append(os.path.join(self.path, file.replace('\n', '')))
