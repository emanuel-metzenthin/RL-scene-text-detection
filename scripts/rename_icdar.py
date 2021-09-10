import os
import shutil

os.mkdir("./new")

for fn in os.listdir("."):
  if not fn.endswith("txt"):
      continue
  num = os.path.splitext(os.path.basename(fn))[0].split("_")[-1]

  shutil.move(fn, f"./new/gt_img_{int(num)-1}.txt")
