import imageio
import torch
import numpy as np


def create_project_w_video(G, outdir, w_steps, images, all_indices, fps=60, device="cuda"):
    G = G.to(device)
    video_project = imageio.get_writer(f'{outdir}/project_w.mp4', mode='I', fps=fps, codec='libx264', bitrate='16M')
    for w_step in w_steps:
        concat_list = []
        for i, index in enumerate(all_indices):
            w = w_step[i].unsqueeze(0).to(device)
            synth_image = render(G, w=w, c=images[index].c_item.c)
            concat_list.append(np.concatenate([images[index].t_uint8, synth_image], axis=0))
        video_project.append_data(np.concatenate(concat_list, axis=1))
    video_project.close()


def create_pti_video(G_steps, outdir, projected_ws, images, all_indices, fps=60, device="cuda"):
    video_pti = imageio.get_writer(f'{outdir}/pti.mp4', mode='I', fps=fps, codec='libx264', bitrate='16M')
    for G in G_steps:
        G.to(device)
        concat_list = []
        for i, w in enumerate(projected_ws):
            w = w.unsqueeze(0).to(device)
            synth_image = render(G, w=w, c=images[all_indices[i]].c_item.c)
            concat_list.append(np.concatenate([images[all_indices[i]].t_uint8, synth_image], axis=0))
        video_pti.append_data(np.concatenate(concat_list, axis=1))
        G.cpu()
    video_pti.close()


def render(G, w, c):
    synth_image = G.synthesis(w, c=c, noise_mode='const')['image']
    synth_image = (synth_image + 1) * (255 / 2)
    synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    return synth_image