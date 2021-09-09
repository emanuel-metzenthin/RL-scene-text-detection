import os
import argparse
import zipfile

parser = argparse.ArgumentParser()
parser.add_argument("path")
args = parser.parse_args()

if not os.path.isdir("./ic15_gt"):
  os.mkdir("./ic15_gt")

zipf = zipfile.ZipFile(f'./ic15_gt/icdar15_gt.zip', 'w', zipfile.ZIP_DEFLATED)

for file_name in os.listdir(args.path):
  if not file_name.endswith(".txt"):
    continue
  file = open(os.path.join(args.path, file_name), "r")
  i = int(os.path.splitext(os.path.basename(file_name))[0].split("_")[-1])
  test_file = open(f'./ic15_gt/gt_img_{i}.txt', 'w+')

  for l in file.readlines():
    bb = l.split(",")[:4]
    box = f"{bb[0]},{bb[1]},{bb[2]},{bb[1]},{bb[2]},{bb[3]},{bb[0]},{bb[3]},\n"
    test_file.write(box)

  test_file.close()
  file.close()
  zipf.write(f'./ic15_gt/gt_img_{i}.txt', arcname=f'gt_img_{i}.txt')

zipf.close()
