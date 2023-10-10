""" Projecting input images into latent spaces. """

import os
import re
import time
from time import perf_counter
import click
import imageio
import numpy as np
import PIL.Image
import torch
import pickle
from tqdm import tqdm
import fnmatch

import dnnlib
import legacy
from inversion.w_inversion import project
from inversion.pti_inversion import project_pti


@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--target', 'target_fname', help='Target image file to project to', required=True, metavar='FILE|DIR')
@click.option('--num-steps', help='Number of optimization steps', type=int, default=500, show_default=True)
@click.option('--num-steps-pti', help='Number of optimization steps for pivot tuning', type=int, default=350,
              show_default=True)
@click.option('--seed', help='Random seed', type=int, default=303, show_default=True)
@click.option('--save-video', help='Save an mp4 video of optimization progress', type=bool, default=True,
              show_default=True)
@click.option('--outdir', help='Where to save the output images', required=True, metavar='DIR')
@click.option('--fps', help='Frames per second of final video', default=30, show_default=True)
@click.option('--num-targets', help='Number of targets to use for inversion', default=10, show_default=True)
def run_projection(
        network_pkl: str,
        target_fname: str,
        outdir: str,
        save_video: bool,
        seed: int,
        num_steps: int,
        num_steps_pti: int,
        fps: int,
        num_targets: int
):
    outdir += ("/" + time.strftime("%Y%m%d-%H%M", time.localtime()))
    outdir += f"_num_targets_{num_targets}"
    pkl_name = network_pkl.split("/")[-1].split(".")[0]
    outdir += f"_{pkl_name}"
    outdir += "_opt_cam"
    os.makedirs(outdir, exist_ok=True)

    np.random.seed(seed)
    torch.manual_seed(seed)

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

    target_uint8 = []
    target_tensor = []
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
        cs.append(c)        # Load target image.
        target_pil = PIL.Image.open(target_fname).convert('RGB')
        w, h = target_pil.size
        s = min(w, h)
        target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        target_pil = target_pil.resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)
        target_pils.append(target_pil)

        t_uint8 = np.array(target_pil, dtype=np.uint8)
        target_uint8.append(t_uint8)
        target_tensor.append(torch.tensor(t_uint8.transpose([2, 0, 1]), device=device))

    # Optimize projection.
    start_time = perf_counter()
    target_indices = list(range(0, len(use_file_names_sorted), len(use_file_names_sorted) // (num_targets - 1)))

    projected_w_steps = project(
        G,
        targets=target_tensor,
        cs=cs,
        num_steps=num_steps,
        device=device,
        verbose=True,
        outdir=outdir,
        target_indices=target_indices
    )
    print(f'Elapsed: {(perf_counter() - start_time):.1f} s')

    G_steps = project_pti(
        G,
        targets=target_tensor,
        w_pivot=projected_w_steps[-1:],
        cs=cs,
        num_steps=num_steps_pti,
        device=device,
        outdir=outdir,
        target_indices=target_indices
    )
    print(f'Elapsed: {(perf_counter() - start_time):.1f} s')

    # Render debug output: optional video and projected image and W vector.
    os.makedirs(outdir, exist_ok=True)
    if save_video:
        video = imageio.get_writer(f'{outdir}/proj.mp4', mode='I', fps=fps, codec='libx264', bitrate='16M')
        print(f'Saving optimization progress video "{outdir}/proj.mp4"')
        for projected_w in projected_w_steps[::2]:
            synth_image = G.synthesis(projected_w.unsqueeze(0).to(device), c=cs[0], noise_mode='const')['image']
            synth_image = (synth_image + 1) * (255 / 2)
            synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
            video.append_data(np.concatenate([target_uint8[0], synth_image], axis=1))
        for G_new in G_steps:
            G_new.to(device)
            synth_image = G_new.synthesis(projected_w_steps[-1].unsqueeze(0).to(device), c=cs[0], noise_mode='const')[
                'image']
            synth_image = (synth_image + 1) * (255 / 2)
            synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
            video.append_data(np.concatenate([target_uint8[0], synth_image], axis=1))
            G_new.cpu()
        video.close()

    # Save final projected frame and W vector.
    target_pils[0].save(f'{outdir}/target.png')
    projected_w = projected_w_steps[-1]
    G_final = G_steps[-1].to(device)
    synth_image = G_final.synthesis(projected_w.unsqueeze(0).to(device), c=cs[0], noise_mode='const')['image']
    synth_image = (synth_image + 1) * (255 / 2)
    synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    PIL.Image.fromarray(synth_image, 'RGB').save(f'{outdir}/proj.png')
    np.savez(f'{outdir}/projected_w.npz', w=projected_w.unsqueeze(0).cpu().numpy())

    with open(f'{outdir}/fintuned_generator.pkl', 'wb') as f:
        network_data["G_ema"] = G_final.eval().requires_grad_(False).cpu()
        pickle.dump(network_data, f)


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    run_projection()

# ----------------------------------------------------------------------------