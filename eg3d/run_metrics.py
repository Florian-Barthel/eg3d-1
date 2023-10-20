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

    print(f"Running metrics on {w_path} with generator {network_pkl}")

    np.random.seed(42)
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metric_helper = Metrics()

    # load data and latent
    checkpoint = np.load(w_path)
    images = load(data_path, 512, device=device)

    for desc, network in [("original_net", original_network), ("PTI_net", network_pkl)]:
        # Load networks.
        print('Loading networks from "%s"...' % network)
        device = torch.device('cuda')
        with dnnlib.util.open_url(network) as fp:
            network_data = legacy.load_network_pkl(fp)
            G = network_data['G_ema'].requires_grad_(False).to(device)
        G.rendering_kwargs["ray_start"] = 2.35

        # setup
        ms_ssim = []
        mse = []
        lpips = []
        id_sim = []
        angles = []

        target_indices = select_evenly(images, num_samples)

        for i in tqdm(target_indices):
            img = images[i]
            if "ws" in checkpoint.keys():
                ws = [torch.tensor(w_).to("cuda") for w_ in checkpoint['ws']]
                cs = [torch.tensor(c_).to("cuda") for c_ in checkpoint['cs']]
                w = torch.tensor(interpolate_w_by_cam(ws, cs, img.c_item.c)).to("cuda")
            else:
                w = torch.tensor(checkpoint["w"]).to("cuda")
            synth_image = G.synthesis(w, c=img.c_item.c, noise_mode='const')['image'][0]

            angles.append(img.c_item.xz_angle().cpu().numpy())
            mse.append(metric_helper.mse(synth_image, img.target_tensor[0]))
            ms_ssim.append(metric_helper.ms_ssim(synth_image, img.target_tensor[0]))
            lpips.append(metric_helper.lpips(synth_image, img.target_tensor[0]).cpu().numpy())
            id_sim.append(metric_helper.id_similarity(synth_image, img.target_tensor[0]))

        ms_ssim = np.array(ms_ssim)
        mse = np.array(mse)
        lpips = np.array(lpips)
        id_sim = np.array(id_sim)
        angles = np.array(angles)

        np.savez(f"{rundir}/{desc}_ms_ssim.npz", values=ms_ssim, mean=np.mean(ms_ssim))
        np.savez(f"{rundir}/{desc}_mse.npz", values=mse, mean=np.mean(mse))
        np.savez(f"{rundir}/{desc}_lpips.npz", values=lpips, mean=np.mean(lpips))
        np.savez(f"{rundir}/{desc}_id_sim.npz", values=id_sim, mean=np.mean(id_sim))
        np.savez(f"{rundir}/angles.npz", values=angles)

        print(rundir)
        print(f"{desc}_mse", np.mean(mse))
        print(f"{desc}_lpips", np.mean(lpips))
        print(f"{desc}_ms_ssim", np.mean(ms_ssim))
        print(f"{desc}_id_sim", np.mean(id_sim))
        print(f"{desc}-------")
        print(f"{np.mean(mse)}\t{np.mean(lpips)}\t{np.mean(ms_ssim)}\t{np.mean(id_sim)}")
        print(f"{desc}-------")


if __name__ == "__main__":
    run()
