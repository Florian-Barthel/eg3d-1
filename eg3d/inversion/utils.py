
from typing import List
import torch
from tqdm import tqdm
import torch.nn.functional as F

import dnnlib


def create_vgg16_features(float_targets: List[torch.tensor], vgg16, verbose=True):
    target_features = []
    float_targets = tqdm(float_targets, desc="Creating Features") if verbose else float_targets
    for target in float_targets:
        target_images_perc = (target + 1) * (255 / 2)
        if target_images_perc.shape[2] > 256:
            target_images_perc = F.interpolate(target_images_perc, size=(256, 256), mode='area')
        target_features.append(vgg16(target_images_perc, resize_images=False, return_lpips=True).detach().cpu())
    return target_features


def convert_float_images(targets: List[torch.tensor], device, verbose=True):
    target_images = []
    targets = tqdm(targets, desc="Creating Images [-1, 1]") if verbose else targets
    for target in targets:
        current_image = target.unsqueeze(0).to(device).to(torch.float32) / 255.0 * 2 - 1
        target_images.append(current_image)
    return target_images


def get_vgg16(device):
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        return torch.jit.load(f).eval().to(device)
