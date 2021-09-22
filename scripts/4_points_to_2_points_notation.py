import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("path")
args = parser.parse_args()

new_path = os.path.join(args.path, "./ic13_gt")
if not os.path.isdir(new_path):
  os.mkdir(new_path)

for file_name in os.listdir(args.path):
  if not file_name.endswith(".txt"):
      continue
  file = open(os.path.join(args.path, file_name), "r")
  i = int(os.path.splitext(os.path.basename(file_name))[0].split("_")[-1])
  test_file = open(f'{new_path}/gt_img_{i}.txt', 'w+')

  for l in file.readlines():
    bb = l.split(",")[:8]
    bb = list(map(int, bb))

    box = f"{min(bb[0], bb[6])}," + \
          f"{min(bb[1], bb[3])}," + \
          f"{max(bb[2], bb[4])}," + \
          f"{max(bb[5], bb[7])},\n"

    test_file.write(box)

  test_file.close()
  file.close()

