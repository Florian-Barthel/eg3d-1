from typing import List
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import torch

from camera_utils import LookAtPoseSampler
from inversion.load_data import ImageItem, CamItem


def create_vgg_features(images: List[ImageItem], vgg, downsampling=True, verbose=True):
    images = tqdm(images, desc="Creating Features") if verbose else images
    for img_item in images:
        img = img_item.target_tensor
        if img.shape[2] > 256 and downsampling:
            img = F.interpolate(img, size=(256, 256), mode='area')
        img_item.feature = vgg(img).detach().cpu()


def create_w_stats(G, w_avg_samples: int, device):
    print(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    camera_lookat_point = torch.tensor([0, 0, 0.0], device=device)
    cam2world_pose = LookAtPoseSampler.sample(3.14 / 2, 3.14 / 2, camera_lookat_point, radius=2.7, device=device)
    intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
    c_samples = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), c_samples.repeat(w_avg_samples, 1))  # [N, L, C]
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)  # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=True)  # [1, 1, C]
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5
    return w_avg, w_std


def interpolate_w_by_cam(ws: List[torch.tensor], cs: List[torch.tensor], c: torch.tensor, device="cuda", verbose=False):
    angle = torch.tensor([CamItem(c).xz_angle()])
    cs = torch.tensor([CamItem(cs[i]).xz_angle() for i in range(len(cs))])

    if angle >= torch.max(cs):
        return ws[-1]

    if angle <= torch.min(cs):
        return ws[0]

    cs_diff = torch.abs(cs - angle)
    closest_index, second_closest_index = torch.argsort(cs_diff)[:2]
    index_left = torch.minimum(closest_index, second_closest_index)
    index_right = torch.maximum(closest_index, second_closest_index)

    total_dist = torch.abs(cs[index_left] - cs[index_right])
    dist_1 = torch.abs(cs[index_left] - angle)
    mag = torch.clip(dist_1 / total_dist, 0, 1).to(device)
    w_int = ws[index_left] * (1 - mag) + ws[index_right] * mag
    if verbose:
        print(f"w{index_left} * {(1 - mag)} + w{index_right} * {mag}")
    return w_int
