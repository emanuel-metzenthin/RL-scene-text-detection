import os
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("path")
args = parser.parse_args()

new_path = os.path.join(args.path, "new")
os.makedirs(new_path, exist_ok=True)
ADD = 3500

for fn in os.listdir(args.path):
  if not fn.endswith(".txt"):
    continue
  name, ext = os.path.splitext(os.path.basename(fn))
  splits = name.split("_")
  prefix = splits[:-1]
  num = int(splits[-1])

  new_file = f"{'_'.join(prefix)}_{num + ADD}{ext}"

  shutil.copy(os.path.join(args.path, fn), os.path.join(new_path, new_file))
