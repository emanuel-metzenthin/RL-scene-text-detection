import json
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("json_file")
args = parser.parse_args()

gt_file = open(args.json_file, 'r')
gt_json = json.loads(gt_file.read())
gt_file.close()

gt_json = np.random.shuffle(gt_json)
num_train = 0.95 * len(gt_json)

train_gt = gt_json[:num_train]
val_gt = gt_json[num_train:]

val_file = open("val_gt.json", "w+")
train_file = open("train_gt.json", "w+")

train_file.write(json.dumps(train_gt))
val_file.write(json.dumps(val_gt))

train_file.close()
val_file.close()
