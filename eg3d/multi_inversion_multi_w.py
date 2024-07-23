import json
import os
import time
import click
import numpy as np
import torch
import pickle
from torch.utils.tensorboard import SummaryWriter

import dnnlib
import legacy
from inversion.multi_w_inversion import project
from inversion.multi_pti_inversion import project_pti


@click.command()
@click.option('--name', required=True, metavar='DIR')
@click.option('--outdir', help='Where to save the output.', required=True, metavar='DIR')
@click.option('--seed', help='Random seed', type=int, default=303, show_default=True)
@click.option('--num-steps', help='Number of optimization steps', type=int, default=1000, show_default=True)
@click.option('--num-steps-pti', help='Number of optimization steps for pivot tuning', type=int, default=500, show_default=True)
@click.option('--downsampling', help='Downsample images from 512 to 256', type=bool, required=False, default=True)
@click.option('--continue-folder', required=True, metavar='FILE')
@click.option('--use-interpolation', type=bool, required=False, default=True)
@click.option('--depth-reg', type=bool, required=False, default=True)
@click.option('--w-norm-reg', type=bool, required=False, default=True)
def run_projection(
        name: str,
        outdir: str,
        seed: int,
        num_steps: int,
        num_steps_pti: int,
        downsampling: bool,
        continue_folder: str,
        use_interpolation: bool,
        depth_reg: bool,
        w_norm_reg: bool
):
    # cur_time = time.strftime("%Y%m%d-%H%M", time.localtime())
    os.makedirs(outdir, exist_ok=True)
    desc = "/" + name
    outdir += desc
    writer = SummaryWriter(outdir)

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load networks.
    network_path = os.path.join(continue_folder, "fintuned_generator.pkl")
    print('Loading networks from "%s"...' % network_path)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_path) as fp:
        network_data = legacy.load_network_pkl(fp)
        G = network_data['G_ema'].requires_grad_(False).to("cpu")

    # Load images
    with open(os.path.join(continue_folder, "images.pkl"), "rb") as input_file:
        images = pickle.load(input_file)
    indices = list(range(len(images)))
    target_indices = indices[::2]
    interpolated_indices = indices[1::2]

    # Load latent vector
    w_checkpoint = np.load(os.path.join(continue_folder, "final_projected_w.npz"))
    w_checkpoint = torch.tensor(w_checkpoint[("w")]).to(device)

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
        w_checkpoint=w_checkpoint,
        use_interpolation=use_interpolation,
        use_depth_reg=depth_reg,
        use_w_norm_reg=w_norm_reg
    )

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

    with open(outdir + "/config.json", "w") as file:
        json.dump({
            "seed": seed,
            "num_steps": num_steps,
            "num_steps_pti": num_steps_pti,
            "downsampling": downsampling,
            "use_interpolation": use_interpolation,
            "continue_folder": continue_folder,
            "depth_reg": depth_reg
        }, file)

    # Save final projected frame and W vector.
    G_final = G_steps[-1].to(device)

    # save results
    with open(f'{outdir}/fintuned_generator.pkl', 'wb') as f:
        network_data["G_ema"] = G_final.eval().requires_grad_(False).cpu()
        pickle.dump(network_data, f)


if __name__ == "__main__":
    run_projection()
