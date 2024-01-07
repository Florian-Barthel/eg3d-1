import copy
import fnmatch
import re
from typing import Tuple, List

from tqdm import tqdm
import torch
import PIL
import numpy as np

import dnnlib


class ImageItem:
    def __init__(self, file_name: str, c: torch.tensor = None, device="cuda", img_resolution=512):
        self.file_name = file_name
        if c is not None:
            self.c_item = CamItem(c)
            self.original_c_item = CamItem(copy.deepcopy(c))
        self.device = device

        # load image.
        self.target_pil = PIL.Image.open(file_name).convert('RGB')
        w, h = self.target_pil.size
        s = min(w, h)
        self.target_pil = self.target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        self.target_pil = self.target_pil.resize((img_resolution, img_resolution), PIL.Image.LANCZOS)
        self.w, self.h = self.target_pil.size
        self.t_uint8 = np.array(self.target_pil, dtype=np.uint8)

        target_tensor = torch.tensor(self.t_uint8.transpose([2, 0, 1]), device=device)
        self.target_tensor = target_tensor.unsqueeze(0).to(torch.float32) / 255.0 * 2 - 1
        self.feature = None


class CamItem:
    def __init__(self, c: torch.tensor):
        self.c = c

    def extrinsic(self):
        return self.c[0, :16].reshape(4, 4)

    def intrinsic(self):
        return self.c[0, 16:].reshape(3, 3)

    def rotation(self):
        return self.extrinsic()[:-1, :-1]

    def xyz(self, as_numpy=False):
        x = self.extrinsic()[0, -1]
        y = self.extrinsic()[1, -1]
        z = self.extrinsic()[2, -1]
        if as_numpy:
            x = x.cpu().detach().numpy()
            y = y.cpu().detach().numpy()
            z = z.cpu().detach().numpy()
        return [x, y, z]

    def xz_angle(self):
        x, y, z = self.xyz()
        angle = torch.arctan2(z, x)
        return angle

    def direction(self):
        direction = torch.matmul(self.rotation().to("cpu"), torch.tensor([[0.0], [0.0], [-1.0]]).to("cpu"))
        mag = torch.sqrt(direction[0, 0]**2 + direction[1, 0]**2 + direction[2, 0]**2)
        return direction / mag


def load(folder: str, img_resolution: int, device="cpu") -> List[ImageItem]:
    dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=folder,
                                     use_labels=True, max_size=None, xflip=False)
    dataset = dnnlib.util.construct_class_by_name(**dataset_kwargs)
    images = []

    use_file_names = sorted(fnmatch.filter(dataset._image_fnames, "[!epoch][!crop]*[0-9].png"))
    # use_file_names_sorted = ["" for _ in range(len(use_file_names))]
    # for file_name in use_file_names:
    #     index = re.findall(r'\d+', file_name)[0]
    #     use_file_names_sorted[int(index)] = file_name
    # use_file_names = use_file_names_sorted

    label_dict = dataset.load_label_dict()

    for filename in tqdm(use_file_names, desc="Loading Data"):
        target_fname = dataset._path + "/" + filename
        c = torch.tensor(label_dict[filename]).to(device)[None, ...]
        images.append(ImageItem(target_fname, c, device, img_resolution))

    return images


if __name__ == "__main__":
    images = load("../../dataset_preprocessing/ffhq/1", 512)
    for img in images:
        print(img.file_name, img.c_item.xz_angle() / torch.pi, *img.c_item.xyz())
