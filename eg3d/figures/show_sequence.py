import glob
import re

import PIL
import click
import numpy as np
import torch
from tqdm import tqdm

import dnnlib
import legacy
from inversion.image_selection import select_evenly
from inversion.load_data import load
from inversion.metrics import Metrics
from inversion.utils import interpolate_w_by_cam


@click.command()
@click.option('--data-path', required=True, metavar='DIR')
@click.option('--num-samples', required=True, type=int)
@click.option('--rundir', required=True, metavar='DIR')
@click.option('--original-network', required=True, metavar='FILE')
def run(
        data_path: str,
        num_samples: int,
        rundir: str,
        original_network: str
):
    network_pkl = rundir + "/fintuned_generator.pkl"
    w_files = glob.glob(rundir + "/*.npz")
    max_num = -1
    max_file = ""
    for w_file in w_files:
        w_file = w_file.split("/")[-1]
        all_nums = re.findall(r'\d+', w_file)
        if len(all_nums) == 0:
            continue
        current_num = int(all_nums[0])
        if current_num > max_num:
            max_num = current_num
            max_file = w_file
    w_path = rundir + "/" + max_file

    np.random.seed(42)
    torch.manual_seed(42)
    device = "cpu"#  if torch.cuda.is_available() else "cpu"

    # load data and latent
    checkpoint = np.load(w_path)
    images = load(data_path, 512, device=device)
    target_indices = select_evenly(images, num_samples)

    # reals
    target_images = []
    for i in tqdm(target_indices):
        img = (images[i].target_tensor[0] + 1) * (255 / 2)
        img = img.clamp(0, 255).to(torch.uint8)
        img = img.permute(1, 2, 0).cpu().numpy()
        target_images.append(img)
    sequence_img = np.concatenate(target_images, axis=1)
    PIL.Image.fromarray(sequence_img, 'RGB').save(f'{rundir}/targets_sequence_{num_samples}.png')

    # fakes
    for desc, network in [("original_net", original_network), ("PTI_net", network_pkl)]:
        print('Loading networks from "%s"...' % network)
        with dnnlib.util.open_url(network) as fp:
            network_data = legacy.load_network_pkl(fp)
            G = network_data['G_ema'].requires_grad_(False).to(device)
        G.rendering_kwargs["ray_start"] = 2.35

        result_images = []
        for i in tqdm(target_indices):
            img = images[i]
            if "ws" in checkpoint.keys():
                ws = [torch.tensor(w_).to(device) for w_ in checkpoint['ws']]
                cs = [torch.tensor(c_).to(device) for c_ in checkpoint['cs']]
                w = torch.tensor(interpolate_w_by_cam(ws, cs, img.c_item.c, device=device)).to(device)
            else:
                w = torch.tensor(checkpoint["w"]).to(device)
            synth_image = G.synthesis(w, c=img.c_item.c, noise_mode='const')['image'][0]
            synth_image = (synth_image + 1) * (255 / 2)
            synth_image = synth_image.clamp(0, 255).to(torch.uint8)
            synth_image = synth_image.permute(1, 2, 0).cpu().numpy()
            result_images.append(synth_image)

        sequence_img = np.concatenate(result_images, axis=1)
        PIL.Image.fromarray(sequence_img, 'RGB').save(f'{rundir}/{desc}_sequence_{num_samples}.png')


if __name__ == "__main__":
    run()
