import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
from typing import List

from inversion.load_data import load
from inversion.load_data import ImageItem


def select_evenly(images: List[ImageItem], num_targets: int):
    all_angles = torch.tensor([item.xz_angle() for item in images])
    min_angle = torch.min(all_angles)
    max_angle = torch.max(all_angles)
    target_angles = torch.linspace(start=min_angle, end=max_angle, steps=num_targets)

    # find the closest match in list and return index
    target_indices = []
    for target_angle in target_angles:
        distance = torch.abs(all_angles - target_angle)
        target_indices.append(torch.argmin(distance))
    return target_indices


if __name__ == "__main__":
    images = load("../../dataset_preprocessing/ffhq/1", 512, device="mps")
    indices = select_evenly(images, 5)
    print(indices)
    print(*[images[i].direction for i in indices])
    print()
