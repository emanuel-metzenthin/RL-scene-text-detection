import json
import os

from dataset.dataset import Dataset


class COCOTextDataset(Dataset):
    def _load_images_and_gt(self):
        gt_file = open(os.path.join(self.json_path, self.split + '.json'))
        gt_json = json.loads(gt_file.read())["anns"]

        self.images = []
        images_anns = {}

        for key, entry in gt_json.items():
            image_id = entry['image_id']
            bbox = list(map(int, entry['bbox']))
            bbox[2] = bbox[0] + bbox[2]
            bbox[3] = bbox[1] + bbox[3]

            if image_id in images_anns:
                images_anns[image_id].append(bbox)
            else:
                images_anns[image_id] = [bbox]
                self.images.append(os.path.join(self.path, f'train2014/COCO_train2014_{"0" * (12 - len(str(image_id)))}{image_id}.jpg'))

        self.gt = list(images_anns.values())
