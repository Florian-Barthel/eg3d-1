import glob
import re

import click
import imageio
import numpy as np
import torch
from tqdm import tqdm

import dnnlib
import legacy
from camera_utils import LookAtPoseSampler, LookAtPoseSamplerCustom
from inversion.image_selection import select_evenly
from inversion.load_data import load
from inversion.utils import interpolate_w_by_cam

def normalize(x):
    return x / torch.sqrt(x.pow(2).sum(-1, keepdim=True))



@click.command()
@click.option('--data-path', required=True, metavar='DIR')
@click.option('--rundir', required=True, metavar='DIR')
def run(
        data_path: str,
        rundir: str,
):
    network_pkl = rundir + "/fintuned_generator.pkl"
    w_files = glob.glob(rundir + "/*.npz")
    max_num = -1
    max_file = ""
    for w_file in w_files:
        w_file = w_file.split("/")[-1]
        w_file = w_file.split("\\")[-1]
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
    device = "cuda"#  if torch.cuda.is_available() else "cpu"

    # load data and latent
    checkpoint = np.load(w_path)
    images = load(data_path, 512, device=device)#[90:]

    with dnnlib.util.open_url(network_pkl) as fp:
        network_data = legacy.load_network_pkl(fp)
        G = network_data['G_ema'].requires_grad_(False).to(device)
    G.rendering_kwargs["ray_start"] = 2.35

    video = imageio.get_writer(f'{rundir}/rotate.mp4', mode='I', fps=30, codec='libx264', bitrate='16M')
    print(f'Saving optimization progress video "{rundir}/rotate.mp4"')
    start_index, end_index = select_evenly(images, 2)
    start_angle = images[start_index].c_item.xz_angle()
    end_angle = images[end_index].c_item.xz_angle()
    position = list(np.linspace(0, 0.9, 135).tolist())

    for mag in tqdm(position[::-1] + position):
        camera_lookat_point = torch.tensor([0, 0, 0.0], device=device)
        cam2world_pose = LookAtPoseSamplerCustom.sample(start_angle + mag * end_angle, 3.14 / 2, camera_lookat_point, radius=2.7, device=device)
        intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
        c_samples = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

        if "ws" in checkpoint.keys():
            ws = [torch.tensor(w_).to(device) for w_ in checkpoint['ws']]
            cs = [torch.tensor(c_).to(device) for c_ in checkpoint['cs']]
            w = torch.tensor(interpolate_w_by_cam(ws, cs, c_samples, device=device)).to(device)
        else:
            w = torch.tensor(checkpoint["w"]).to(device)

        synth_image = G.synthesis(w.to(device), c=c_samples, noise_mode='const')['image']
        synth_image = (synth_image + 1) * (255 / 2)
        synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
        # video.append_data(np.concatenate([image.t_uint8, synth_image], axis=0))
        video.append_data(synth_image)
    video.close()


if __name__ == "__main__":
    run()
