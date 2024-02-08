
import os
import glob
import PIL.Image as Image
import numpy as np

from tqdm import tqdm

img_path = "../../dataset_preprocessing/ffhq/1_stabilize"
out_path = "../../dataset_preprocessing/ffhq/1"
os.makedirs(out_path, exist_ok=True)

all_images = sorted(glob.glob(img_path + "/*.jpg"))[90:]
x_shift = 0.0
for img in tqdm(all_images):
    img_pil = Image.open(img).convert('RGB')
    size = 512
    crop_img = img_pil.crop((int(np.round(x_shift)) + 300, 100, int(np.round(x_shift)) + 700, 500))
    x_shift += 1.5
    resize_img = crop_img.resize((size, size))
    saf = os.path.basename(img)
    resize_img.save(os.path.join(out_path, os.path.basename(img)))


