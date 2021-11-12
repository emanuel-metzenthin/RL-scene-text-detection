import os
from dataset.dataset import Dataset


class LSVTDataset(Dataset):
    def _load_images_and_gt(self):
        file_names = [i for i in os.listdir(self.path) if i.lower()[-4:] in ['.jpg', '.png', '.gif']]
        file_names.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))
        self.images = [os.path.join(self.path, i) for i in file_names]
        self.gt = [[] for _ in self.images]