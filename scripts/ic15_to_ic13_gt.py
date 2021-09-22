import os
import argparse
import zipfile

parser = argparse.ArgumentParser()
parser.add_argument("path")
args = parser.parse_args()

if not os.path.isdir("./ic13_gt"):
  os.mkdir("./ic13_gt")

zipf = zipfile.ZipFile(f'./ic13_gt/icdar13_gt.zip', 'w', zipfile.ZIP_DEFLATED)

for file_name in os.listdir(args.path):
  if not file_name.endswith(".txt"):
    continue
  file = open(os.path.join(args.path, file_name), "r")
  i = int(os.path.splitext(os.path.basename(file_name))[0].split("_")[-1])
  test_file = open(f'./ic13_gt/gt_img_{i}.txt', 'w+')

  for l in file.readlines():
    bb = list(map(int, l.split(",")[:8]))
    x_values = [bb[x] for x in range(len(bb)) if x % 2 == 0]
    y_values = [bb[y] for y in range(len(bb)) if y % 2 == 1]

    box = f"{min(x_values)}," + \
          f"{min(y_values)}," + \
          f"{max(x_values)}," + \
          f"{max(y_values)},\n"

    test_file.write(box)

  test_file.close()
  file.close()
  zipf.write(f'./ic13_gt/gt_img_{i}.txt', arcname=f'gt_img_{i}.txt')

zipf.close()
