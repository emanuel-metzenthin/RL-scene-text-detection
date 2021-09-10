import json
import os
import numpy as np
from dataset.dataset import Dataset


class SimpleDataset(Dataset):
    def _load_images_and_gt(self):
        self.images = []
        self.gt = []
        img_files = open(os.path.join(self.path, 'image_locations.txt')).readlines()
        gt_arr = np.load(os.path.join(self.path, 'bounding_boxes.npy'), allow_pickle=True)

        for file in img_files:
            self.images.append(os.path.join(self.path, file.replace('\n', '')))

        for img_gt in gt_arr:
            bboxes = []
            for b in img_gt:
                bboxes.append([x for t in zip(*b) for x in t])
            self.gt.append(img_gt)
