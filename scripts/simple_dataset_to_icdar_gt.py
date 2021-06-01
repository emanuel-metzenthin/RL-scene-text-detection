import argparse
import os
import zipfile

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("data_path")
args = parser.parse_args()

bboxes = np.load(os.path.join(args.data_path, "bounding_boxes.npy"))
os.mkdir("./simple_gt")
zipf = zipfile.ZipFile('./simple_gt/simple_gt.zip', 'w', zipfile.ZIP_DEFLATED)

for i, bb_list in enumerate(bboxes):
    test_file = open(f'./simple_gt/gt_img_{i}.txt', 'w+')

    for bb in bb_list:
        box = f"{bb[0]},{bb[0][1]},{bb[1][0]},{bb[1][1]},\n"
        test_file.write(box)

    test_file.close()
    zipf.write(f'./simple_gt/gt_img_{i}.txt', arcname=f'res_img_{i}.txt')

zipf.close()