import os
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("path")
args = parser.parse_args()

ADD = 1000

for fn in os.listdir(args.path):
  name, ext = os.path.splitext(os.path.basename(fn))
  splits = name.split("_")
  prefix = splits[:-1]
  num = int(splits[-1])

  new_file = f"{'_'.join(prefix)}_{num + ADD}.{ext}"

  shutil.move(os.path.join(args.path, fn), os.path.join(args.path, new_file))
