import json
import os
import re
from typing import Text

from dataset.dataset import Dataset


class LSVTIcdarMixDataset(Dataset):

    def __init__(self, path: Text, json_path: Text, mix_path, mix_labels, split: Text = 'train', img_size=(224, 224)):
        self.mix_path = mix_path
        self.mix_labels = mix_labels

        super().__init__(path, json_path, split, img_size)

    def _load_lsvt(self):
        file_names = [i for i in os.listdir(self.path) if i.lower()[-4:] in ['.jpg', '.png', '.gif']]
        file_names.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))
        images = [os.path.join(self.path, i) for i in file_names]
        gt = [[] for _ in range(len(images))]

        return images, gt

    def _load_icdar_gt(self):
        folder = os.path.join(self.mix_path, self.split + '_images')
        file_names = [i for i in os.listdir(folder) if i.lower()[-4:] in ['.jpg', '.png', '.gif']]
        file_names.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))
        images = [os.path.join(folder, i) for i in file_names]
        gt = []

        for file_name in file_names:
            img_name, _ = os.path.splitext(file_name)
            file = open(os.path.join(self.mix_path, self.split + '_gt', 'gt_' + img_name + '.txt'))
            img_gt = []

            for line in file.readlines():
                if line.isspace():
                    continue
                line = re.sub('".*?"', '', line)

                if ', ' in line:
                    sep = ', '
                elif ',' in line:
                    sep = ','
                else:
                    sep = ' '
                x1, y1, x2, y2 = line.split(sep)[:4]
                img_gt.append((float(x1), float(y1), float(x2), float(y2)))

            gt.append(img_gt)

        return images, gt

    def _load_images_and_gt(self):
        self.images = []
        self.gt = []

        imgs, gt = self._load_lsvt()
        self.images += imgs
        self.gt += gt

        imgs, gt = self._load_icdar_gt()
        self.images += imgs
        self.gt += gt
