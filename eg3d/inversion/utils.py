from typing import List
from tqdm import tqdm
import torch.nn.functional as F

from inversion.load_data import ImageItem


def create_vgg_features(images: List[ImageItem], vgg, downsampling=True, verbose=True):
    images = tqdm(images, desc="Creating Features") if verbose else images
    for img_item in images:
        img = img_item.target_tensor
        if img.shape[2] > 256 and downsampling:
            img = F.interpolate(img, size=(256, 256), mode='area')
        img_item.feature = vgg(img).detach().cpu()

