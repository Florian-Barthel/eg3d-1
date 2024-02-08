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
from utils.run_dir import get_pkl_and_w


@click.command()
@click.option('--data-path', required=True, metavar='DIR')
@click.option('--rundir', required=True, metavar='DIR')
@click.option('--depth-samples', required=True, type=int)
def run(
        data_path: str,
        rundir: str,
        depth_samples: int
):
    network_pkl, w_path = get_pkl_and_w(rundir)
    np.random.seed(42)
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load data and latent
    checkpoint = np.load(w_path)
    if "ws" not in checkpoint.keys():
        print("Checkpoint has no multiple ws")
        return
    ws = [torch.tensor(w_).to("cuda") for w_ in checkpoint['ws']]
    cs = [torch.tensor(c_).to("cuda") for c_ in checkpoint['cs']]

    images = load(data_path, 512, device=device)
    depth_target_indices = select_evenly(images, depth_samples)

    with dnnlib.util.open_url(network_pkl) as fp:
        network_data = legacy.load_network_pkl(fp)
        G = network_data['G_ema'].requires_grad_(False).to(device)
    G.rendering_kwargs["ray_start"] = 2.35

    depth_metric = DepthMetric(depth_samples)

    for i in tqdm(depth_target_indices):
        img = images[i]
        for j in range(depth_samples):
            current_cam = images[j].c_item.c
            w = torch.tensor(interpolate_w_by_cam(ws, cs, img.c_item.c)).to(device)
            depth_image = G.synthesis(w, c=current_cam, noise_mode='const')['image_depth'][0]
            depth_metric.update(j, depth_image)

    depth_mean_std = depth_metric.calc_mean_stddev(path=rundir + "/" + "depth.png")
    print(f"depth_mean_std PTI: {depth_mean_std}")


if __name__ == "__main__":
    run()
