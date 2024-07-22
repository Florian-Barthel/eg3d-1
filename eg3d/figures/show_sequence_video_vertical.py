import click
import imageio
import numpy as np
import torch
from tqdm import tqdm

import dnnlib
import legacy
from camera_utils import LookAtPoseSamplerCustom
from inversion.utils import interpolate_w_by_cam
from utils.run_dir import get_pkl_and_w


@click.command()
@click.option('--rundir', required=True, metavar='DIR')
def run(rundir: str):
    network_pkl, w_path = get_pkl_and_w(rundir, verbose=True)
    np.random.seed(42)
    torch.manual_seed(42)
    device = "cuda"
    checkpoint = np.load(w_path)

    with dnnlib.util.open_url(network_pkl) as fp:
        network_data = legacy.load_network_pkl(fp)
        G = network_data['G_ema'].requires_grad_(False).to(device)
    G.rendering_kwargs["ray_start"] = 2.35

    frames = []
    gif_path = f'{rundir}/rotate_vertical.gif'
    position = list(np.linspace(0, 2 * np.pi, 300).tolist())
    intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)

    for mag in tqdm(position):
        camera_lookat_point = torch.tensor([0, 0, 0.0], device=device)
        x = np.cos(mag) * 0.9
        y = np.sin(mag) * 0.4
        cam2world_pose = LookAtPoseSamplerCustom.sample(np.pi / 2 + x, np.pi / 2 + y, camera_lookat_point, radius=2.7, device=device)
        c_samples = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

        if "ws" in checkpoint.keys():
            ws = [torch.tensor(w_).to(device) for w_ in checkpoint['ws']]
            cs = [torch.tensor(c_).to(device) for c_ in checkpoint['cs']]
            w = torch.tensor(interpolate_w_by_cam(ws, cs, c_samples)).to(device)
        else:
            w = torch.tensor(checkpoint["w"]).to(device)

        synth_image = G.synthesis(w.to(device), c=c_samples, noise_mode='const')['image']
        synth_image = (synth_image + 1) * (255 / 2)
        synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
        frames.append(synth_image)
    imageio.mimsave(gif_path, frames, fps=30)


if __name__ == "__main__":
    run()
