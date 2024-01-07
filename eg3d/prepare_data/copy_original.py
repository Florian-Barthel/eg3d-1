
import os
import re

import cv2
import shutil
import glob

from tqdm import tqdm

img_path = "../../dataset_preprocessing/ffhq/1"
out_path = "../../dataset_preprocessing/ffhq/1_stabilize"
os.makedirs(out_path, exist_ok=True)

all_images = glob.glob(img_path + "/*.jpg")
for img in tqdm(all_images):
    old_name = os.path.basename(img)
    index = int(re.findall(r'\d+', old_name)[0])
    new_name = "frame" + str(index).zfill(5) + ".jpg"
    shutil.copy(img, out_path + "/" + new_name)


