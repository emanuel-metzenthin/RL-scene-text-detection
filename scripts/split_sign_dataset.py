import json
import argparse
import numpy as np
import ijson

parser = argparse.ArgumentParser()
parser.add_argument("json_file")
args = parser.parse_args()

gt_file = open(args.json_file, 'r')
gt_json = ijson.items(gt_file)
gt_file.close()

train_gt = []
val_gt = []

for obj in gt_json:
    if np.random.rand() < 0.8:
        train_gt.append(obj)
    else:
        val_gt.append(obj)

val_file = open("val_gt.json", "w+")
train_file = open("train_gt.json", "w+")

train_file.write(json.dumps(train_gt))
val_file.write(json.dumps(val_gt))

train_file.close()
val_file.close()
