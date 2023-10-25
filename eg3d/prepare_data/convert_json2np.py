import numpy as np
import json

image_folder = "../../data/1/"


with open(image_folder + "dataset.json", "r") as f:
    camera_dict = json.load(f)["labels"]
    for name, params in camera_dict:
        np.save(image_folder + name[:-len(".jpg")] + ".npy", np.array(params))