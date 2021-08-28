import os
import json
import re
from dataset.dataset import Dataset


class ICDARDataset(Dataset):
    def _load_images_and_gt(self):
        folder = os.path.join(self.path, self.split + '_images')
        file_names = [i for i in os.listdir(folder) if i.lower().endswith('.jpg')]
        self.images = [os.path.join(folder, i) for i in file_names]
        self.gt = []

        for file_name in file_names:
            file = open(os.path.join(self.path, self.split + '_gt', 'gt_' + file_name.replace('.jpg', '.txt')))
            gt = []

            for line in file.readlines():
                if line.isspace():
                    continue
                line = re.sub('".*?"', '', line)

                if ', ' in line:
                    sep = ', '
                elif ',' in line:
                    sep= ','
                else: 
                    sep = ' '
                x1, y1, x2, y2 = line.split(sep)[:4]
                gt.append((float(x1), float(y1), float(x2), float(y2)))

            self.gt.append(gt)
