import os
import shutil
from tqdm import tqdm

src_path = "C:/Users/flori/Desktop/3DIL_dataset_filter_conf_all"
dest_path = "C:/Users/flori/Desktop/3DIL_dataset_filter_conf"

for folder in tqdm(os.listdir(src_path)):
    shutil.copytree(os.path.join(src_path, folder, "dataset"), os.path.join(dest_path, folder))
