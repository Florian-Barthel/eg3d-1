import os
import numpy as np
import json

image_folder = "../../dataset_preprocessing/ffhq/"
dest_folder = "../../data/"


for i in range(6):
    current_image_folder = f"{image_folder}/{i +1 }/"
    current_dest_folder = f"{dest_folder}/{i +1 }/"
    with open(current_image_folder + "dataset.json", "r") as f:
        camera_dict = json.load(f)["labels"]
        os.makedirs(current_dest_folder, exist_ok=True)
        for name, params in camera_dict:
            np.save(current_dest_folder + name[:-len(".jpg")] + ".npy", np.array(params))
