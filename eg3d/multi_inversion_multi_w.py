import json
import os
import time
from time import perf_counter
from typing import List
import click
import imageio
import numpy as np
import PIL.Image
import torch
import pickle
from torch.utils.tensorboard import SummaryWriter

import dnnlib
import legacy
from inversion.multi_w_inversion import project
from inversion.multi_pti_inversion import project_pti
from inversion.load_data import ImageItem, load
from inversion.image_selection import select_evenly_interpolate, select_evenly
from inversion.utils import interpolate_w_by_cam
from inversion.video import create_project_w_video, create_pti_video


@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--target', 'target_fname', help='Target image file to project to', required=True, metavar='FILE|DIR')
@click.option('--num-steps', help='Number of optimization steps', type=int, default=500, show_default=True)
@click.option('--num-steps-pti', help='Number of optimization steps for pivot tuning', type=int, default=500, show_default=True)
@click.option('--seed', help='Random seed', type=int, default=303, show_default=True)
@click.option('--save-video', help='Save an mp4 video of optimization progress', type=bool, default=True, show_default=True)
@click.option('--outdir', help='Where to save the output.', required=True, metavar='DIR')
@click.option('--num-targets', help='Number of targets to use for inversion', default=10, show_default=True)
@click.option('--downsampling', help='Downsample images from 512 to 256', type=bool, required=True)
@click.option('--continue-w', help='numpy .npz file to load the latent vector', required=True, metavar='FILE')
@click.option('--use-interpolation', type=bool, required=True)
@click.option('--depth-reg', type=bool, required=True)
@click.option('--w-norm-reg', type=bool, required=False, default=True)
def run_projection(
        network_pkl: str,
        target_fname: str,
        outdir: str,
        save_video: bool,
        seed: int,
        num_steps: int,
        num_steps_pti: int,
        num_targets: int,
        downsampling: bool,
        continue_w: str,
        use_interpolation: bool,
        depth_reg: bool,
        w_norm_reg: bool
):
    # cur_time = time.strftime("%Y%m%d-%H%M", time.localtime())
    # desc = ("/" + cur_time)
    desc = "/multi-w"
    # desc += f"_targets_{num_targets}"
    # desc += f"_iter_{num_steps}_{num_steps_pti}"
    # desc += "_inter" if use_interpolation else ""
    # desc += "_depth_reg" if depth_reg else ""
    # desc += "_without_norm_reg" if not w_norm_reg else ""
    data_index = target_fname.split("/")[-1]
    desc += f"_{data_index}"
    os.makedirs(outdir, exist_ok=True)
    outdir += desc
    writer = SummaryWriter(outdir)

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load networks.
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as fp:
        network_data = legacy.load_network_pkl(fp)
        G = network_data['G_ema'].requires_grad_(False).to("cpu")  # type: ignore
    G.rendering_kwargs["ray_start"] = 2.35

    images: List[ImageItem] = load(target_fname, img_resolution=G.img_resolution, device=device)

    start_time = perf_counter()
    all_indices = select_evenly(images, num_targets + num_targets - 1)
    target_indices, interpolated_indices = select_evenly_interpolate(images, num_targets)

    projected_w_steps = project(
        G,
        images=images,
        num_steps=num_steps,
        device=device,
        outdir=outdir,
        target_indices=target_indices,
        inter_indices=interpolated_indices,
        writer=writer,
        downsampling=downsampling,
        continue_checkpoint=continue_w,
        use_interpolation=use_interpolation,
        use_depth_reg=depth_reg,
        use_w_norm_reg=w_norm_reg
    )
    time_opt_w = (perf_counter() - start_time)
    start_time = perf_counter()

    all_ws = [[] for _ in range(len(projected_w_steps))]
    target_cams = [images[target_index].c_item.c for target_index in target_indices]
    for i, w_step in enumerate(projected_w_steps):
        for j in all_indices:
            target_cam = images[j].c_item.c
            w = interpolate_w_by_cam(w_step, target_cams, target_cam)
            all_ws[i].append(w)

    if save_video:
        create_project_w_video(G=G, outdir=outdir, w_steps=all_ws, images=images, all_indices=all_indices)

    G_steps = project_pti(
        G,
        images=images,
        w_pivots=projected_w_steps[-1],
        num_steps=num_steps_pti,
        device=device,
        outdir=outdir,
        target_indices=target_indices,
        inter_indices=interpolated_indices,
        writer=writer,
        downsampling=downsampling,
        use_interpolation=use_interpolation,
        use_depth_reg=depth_reg
    )
    time_opt_pti = (perf_counter() - start_time)

    with open(outdir + "/config.json", "w") as file:
        json.dump({
            "net": network_pkl,
            "target_fname": target_fname,
            "seed": seed,
            "num_steps": num_steps,
            "num_steps_pti": num_steps_pti,
            "num_targets": num_targets,
            "downsampling": downsampling,
            # "time": cur_time,
            "time_w": time_opt_w,
            "time_pti": time_opt_pti,
            "use_interpolation": use_interpolation,
            "continue_w": continue_w,
            "depth_reg": depth_reg
        }, file)

    # Save final projected frame and W vector.
    G_final = G_steps[-1].to(device)
    for i in range(len(all_indices)):
        target_img = images[all_indices[i]]
        target_img.target_pil.save(f'{outdir}/target_{i}.png')
        target_cam = target_img.c_item.c
        w = all_ws[-1][i]
        synth_image = G_final.synthesis(w.unsqueeze(0).to(device), c=target_cam, noise_mode='const')['image']
        synth_image = (synth_image + 1) * (255 / 2)
        synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
        PIL.Image.fromarray(synth_image, 'RGB').save(f'{outdir}/proj_{i}.png')

    # save results
    np.savez(f'{outdir}/projected_w.npz', w=projected_w_steps[-1].cpu().numpy())
    with open(f'{outdir}/fintuned_generator.pkl', 'wb') as f:
        network_data["G_ema"] = G_final.eval().requires_grad_(False).cpu()
        pickle.dump(network_data, f)

    # create video
    G = G.to(device)
    if save_video:
        create_pti_video(G_steps=G_steps, outdir=outdir, projected_ws=all_ws[-1], images=images, all_indices=all_indices)


if __name__ == "__main__":
    run_projection()
