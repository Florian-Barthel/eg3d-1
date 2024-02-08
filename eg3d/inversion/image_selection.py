import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
from typing import List
from inversion.load_data import load
from inversion.load_data import ImageItem


def select_evenly(images: List[ImageItem], num_targets: int):
    """
    Given a list of images with camera parameters (c_item), this function returns the indices of target images that are
    evenly distributed from the lowest angle xz angle to the highest.
    """
    all_angles = torch.tensor([item.c_item.xz_angle() for item in images])
    min_angle = torch.min(all_angles)
    max_angle = torch.max(all_angles)

    if num_targets == 1:
        # select the most centered view
        target_angles = [torch.pi / 2]
    else:
        target_angles = torch.linspace(start=min_angle, end=max_angle, steps=num_targets)

    # find the closest match in list and return index
    target_indices = []
    for target_angle in target_angles:
        distance = torch.abs(all_angles - target_angle)
        target_indices.append(torch.argmin(distance))
    return target_indices


def select_evenly_interpolate(images: List[ImageItem], num_targets: int):
    """
    Calls select_evenly and then splits the list of indices into two lists with alternating values.
    num_targets=7: a, b, a, b, a, b, a -> target_indices = [a,a,a,a], interpolated_indices = [b,b,b]
    """
    num_total = num_targets + num_targets - 1
    all_indices = select_evenly(images, num_total)
    target_indices = []
    interpolated_indices = []
    for i in range(len(all_indices)):
        if i % 2 == 0 or i == len(all_indices) - 1:
            target_indices.append(all_indices[i])
        else:
            interpolated_indices.append(all_indices[i])
    return target_indices, interpolated_indices


if __name__ == "__main__":
    for i in range(6):
        images = load(f"../../dataset_preprocessing/ffhq/{i + 1}", 512)
        target_indices = select_evenly(images, 1)
        print(i + 1, target_indices)
