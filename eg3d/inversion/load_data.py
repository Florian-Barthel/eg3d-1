import fnmatch
import re
from tqdm import tqdm
import torch
import PIL
import numpy as np

import dnnlib


class ImageItem:
    def __init__(self, file_name: str, c: torch.tensor, device, img_resolution=512):
        self.file_name = file_name
        self.c = c
        self.device = device

        # extrinsic parameters
        self.extrinsic = c[0, :16].reshape(4, 4)
        self.rotation = self.extrinsic[:-1, :-1]
        self.x = self.extrinsic[0, -1]
        self.y = self.extrinsic[1, -1]
        self.z = self.extrinsic[2, -1]
        self.xz_angle = torch.arctan2(self.z, self.x)
        self.direction = torch.matmul(self.rotation.to("cpu"), torch.tensor([[0.0], [0.0], [-1.0]]).to("cpu"))
        mag = torch.sqrt(self.direction[0, 0]**2 + self.direction[1, 0]**2 + self.direction[2, 0]**2)
        self.direction /= mag

        # intrinsic parameters
        self.intrinsic = c[0, 16:].reshape(3, 3)
        self.focal_x = self.intrinsic[0, 0]
        self.focal_y = self.intrinsic[1, 1]

        # load image.
        self.target_pil = PIL.Image.open(file_name).convert('RGB')
        w, h = self.target_pil.size
        s = min(w, h)
        self.target_pil = self.target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        self.target_pil = self.target_pil.resize((img_resolution, img_resolution), PIL.Image.LANCZOS)
        self.w, self.h = self.target_pil.size
        self.t_uint8 = np.array(self.target_pil, dtype=np.uint8)
        self.target_tensor = torch.tensor(self.t_uint8.transpose([2, 0, 1]), device=device)


def load(folder: str, img_resolution: int, device="cpu"):
    dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=folder,
                                     use_labels=True, max_size=None, xflip=False)
    dataset = dnnlib.util.construct_class_by_name(**dataset_kwargs)
    images = []

    use_file_names = sorted(fnmatch.filter(dataset._image_fnames, "[!crop]*[0-9].png"))
    use_file_names_sorted = ["" for _ in range(len(use_file_names))]
    for file_name in use_file_names:
        index = re.findall(r'\d+', file_name)[0]
        use_file_names_sorted[int(index)] = file_name
    label_dict = dataset.load_label_dict()

    for idx in tqdm(range(len(use_file_names_sorted)), desc="Loading Data"):
        target_fname = dataset._path + "/" + use_file_names_sorted[idx]
        c = torch.tensor(label_dict[use_file_names_sorted[idx]]).to(device)[None, ...]
        images.append(ImageItem(target_fname, c, device, img_resolution))

    return images
