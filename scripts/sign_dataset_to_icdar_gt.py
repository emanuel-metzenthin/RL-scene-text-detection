import argparse
import json
import os
import zipfile

parser = argparse.ArgumentParser()
parser.add_argument("json_file")
parser.add_argument("--type", default="ic13")
args = parser.parse_args()

file = open(args.json_file, 'r')
gt = json.loads(file.read())
file.close()

os.mkdir("./sign_gt")
zipf = zipfile.ZipFile(f'./sign_gt/sign_gt_{args.type}.zip', 'w', zipfile.ZIP_DEFLATED)

c = 0

for i, dict in enumerate(gt):
    test_file = open(f'./sign_gt/gt_img_{i}.txt', 'w+')
    bb_list = dict["bounding_boxes"]

    for bb in bb_list:
        if bb[0] >= bb[2] or bb[1] >= bb[3] or any([b < 0 for b in bb]):
            c += 1
            continue

        if args.type == 'ic13':
            box = f"{','.join(map(str, bb))},\n"
        else:
            box = f"{bb[0]},{bb[1]},{bb[2]},{bb[1]},{bb[2]},{bb[3]},{bb[0]},{bb[3]},\n"
        test_file.write(box)

    test_file.close()
    zipf.write(f'./sign_gt/gt_img_{i}.txt', arcname=f'gt_img_{i}.txt')
print(f"count: {c}")
zipf.close()
