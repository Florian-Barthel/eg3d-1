import glob
import re
import click
import numpy as np
import torch
from tqdm import tqdm

import dnnlib
import legacy
from inversion.image_selection import select_evenly
from inversion.load_data import load
from inversion.metrics import DepthMetric
from inversion.utils import interpolate_w_by_cam


@click.command()
@click.option('--data-path', required=True, metavar='DIR')
@click.option('--rundir', required=True, metavar='DIR')
@click.option('--depth-samples', required=True, type=int)
def run(
        data_path: str,
        rundir: str,
        depth_samples: int
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

    print(f"Running metrics on {w_path} with generator {network_pkl}")

    np.random.seed(42)
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load data and latent
    checkpoint = np.load(w_path)
    if "ws" not in checkpoint.keys():
        print("Checkpoint has no multiple ws")
        return
    images = load(data_path, 512, device=device)

    # Load networks.
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as fp:
        network_data = legacy.load_network_pkl(fp)
        G = network_data['G_ema'].requires_grad_(False).to(device)
    G.rendering_kwargs["ray_start"] = 2.35


    depth_metric = DepthMetric(depth_samples)
    depth_target_indices = select_evenly(images, depth_samples)

    for i in tqdm(depth_target_indices):
        img = images[i]
        for j in range(depth_samples):
            current_cam = images[j].c_item.c
            ws = [torch.tensor(w_).to("cuda") for w_ in checkpoint['ws']]
            cs = [torch.tensor(c_).to("cuda") for c_ in checkpoint['cs']]
            w = torch.tensor(interpolate_w_by_cam(ws, cs, img.c_item.c)).to("cuda")
            depth_image = G.synthesis(w, c=current_cam, noise_mode='const')['image_depth'][0]
            depth_metric.update(j, depth_image)

    depth_mean_std = depth_metric.calc_mean_stddev(path=rundir + "/" + "depth.png")
    print(f"depth_mean_std PTI: {depth_mean_std}")


if __name__ == "__main__":
    run()
