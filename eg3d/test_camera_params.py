""" Projecting input images into latent spaces. """

import os
import re
import time
import click
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm
import fnmatch

import dnnlib
import legacy
from camera_utils import LookAtPoseSampler


@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--target', 'target_fname', help='Target image file to project to', required=True, metavar='FILE|DIR')
@click.option('--outdir', help='Where to save the output images', required=True, metavar='DIR')
@click.option('--num-targets', help='Number of targets to use for inversion', default=10, show_default=True)
def vis_cam(
        network_pkl: str,
        target_fname: str,
        outdir: str,
        num_targets: int
):
    outdir += ("/" + time.strftime("%Y%m%d-%H%M", time.localtime()))
    outdir += f"_test_cams"
    os.makedirs(outdir, exist_ok=True)

    # Load networks.
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as fp:
        network_data = legacy.load_network_pkl(fp)
        G = network_data['G_ema'].requires_grad_(False).to(device)  # type: ignore

    G.rendering_kwargs["ray_start"] = 2.35

    dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=target_fname,
                                     use_labels=True, max_size=None, xflip=False)
    dataset = dnnlib.util.construct_class_by_name(**dataset_kwargs)  # Subclass of training.dataset.Dataset.

    target_pils = []
    cs = []


    use_file_names = sorted(fnmatch.filter(dataset._image_fnames, "[!crop]*[0-9].png"))
    use_file_names_sorted = ["" for _ in range(len(use_file_names))]
    for file_name in use_file_names:
        index = re.findall(r'\d+', file_name)[0]
        use_file_names_sorted[int(index)] = file_name
    label_dict = dataset.load_label_dict()

    for idx in tqdm(range(len(use_file_names_sorted)), desc="Loading Data"):
        target_fname = dataset._path + "/" + use_file_names_sorted[idx]
        c = torch.tensor(label_dict[use_file_names_sorted[idx]]).to(device)[None, ...]
        cs.append(c)
        # Load target image.
        target_pil = PIL.Image.open(target_fname).convert('RGB')
        w, h = target_pil.size
        s = min(w, h)
        target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        target_pil = target_pil.resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)
        target_pils.append(target_pil)

    # Optimize projection.
    target_indices = list(range(0, len(use_file_names_sorted), len(use_file_names_sorted) // (num_targets - 1)))

    z_samples = np.random.RandomState(123).randn(1, G.z_dim)
    camera_lookat_point = torch.tensor([0, 0, 0.0], device=device)
    cam2world_pose = LookAtPoseSampler.sample(3.14 / 2, 3.14 / 2, camera_lookat_point, radius=2.7, device=device)
    intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
    c_samples = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    w = G.mapping(torch.from_numpy(z_samples).to(device), c_samples[None, ...])  # [N, L, C]

    for i in target_indices:
        synth_image = G.synthesis(w.to(device), c=cs[i], noise_mode='const')['image']
        synth_image = (synth_image + 1) * (255 / 2)
        synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
        synth_image = np.concatenate([synth_image, target_pils[i]])
        PIL.Image.fromarray(synth_image, 'RGB').save(f'{outdir}/{i}.png')



# ----------------------------------------------------------------------------

if __name__ == "__main__":
    vis_cam()

# ----------------------------------------------------------------------------