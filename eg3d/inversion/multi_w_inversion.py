import copy
import os
from pathlib import Path
from typing import List
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from inversion.utils import create_vgg_features, create_w_stats, interpolate_w_by_cam
from inversion.loss import perc, mse, IDLoss, DepthLossAll
from inversion.custom_vgg import CustomVGG, NvidiaVGG16
from inversion.load_data import ImageItem


def project(
        G,
        images: List[ImageItem],
        *,
        num_steps=1000,
        w_avg_samples=10000,
        initial_learning_rate=0.01, # 0.1
        lr_rampdown_length=0.25,
        lr_rampup_length=0.05,
        device: torch.device,
        outdir: str,
        target_indices: List[int],
        inter_indices: List[int],
        downsampling=True,
        writer: SummaryWriter,
        w_checkpoint,
        use_interpolation,
        use_depth_reg: bool,
        use_w_norm_reg: bool
):
    assert images[0].target_tensor[0].shape == (G.img_channels, G.img_resolution, G.img_resolution)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device)
    _, w_std = create_w_stats(G, w_avg_samples, device)

    vgg = CustomVGG("vgg16").to(device)
    create_vgg_features(images, vgg, downsampling=downsampling)
    id_loss_model = IDLoss()
    depth_loss_model = DepthLossAll(num_targets=len(target_indices), depth_multiplier=1)

    w_opt_list = []
    for _ in range(len(target_indices)):
        w_opt = w_checkpoint.detach().clone()
        w_opt.requires_grad = True
        w_opt_list.append(w_opt)

    w_out = torch.zeros([num_steps, len(target_indices)] + list(w_opt_list[0].shape[1:]), dtype=torch.float32, device="cpu")
    optimizer = torch.optim.Adam(w_opt_list, betas=(0.9, 0.999), lr=initial_learning_rate)


    for step in tqdm(range(num_steps)):
        # Learning rate schedule.
        t = step / num_steps
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # normal step
        agg_mse_loss = 0
        agg_perc_loss = 0
        agg_w_norm_loss = 0
        agg_loss = 0
        agg_id_loss = 0
        for count, pair in enumerate(zip(target_indices, w_opt_list)):
            i, w_opt = pair
            synth = G.synthesis(w_opt, c=images[i].c_item.c, noise_mode='const')
            synth_image = synth['image']
            perc_loss = perc(images[i].feature, synth_image, vgg=vgg, downsampling=downsampling)
            # perc_loss_2 = perc(images[i].feature, images[i].target_tensor, vgg=vgg, downsampling=downsampling)
            mse_loss = mse(images[i].target_tensor, synth_image)
            w_norm_loss = 0
            if use_w_norm_reg:
                w_norm_loss = mse(w_opt, w_checkpoint)
            id_loss = id_loss_model(synth_image=synth_image, target_image=images[i].target_tensor)
            loss = 0.1 * mse_loss + perc_loss + 1.0 * w_norm_loss + id_loss
            loss.backward()
            agg_mse_loss += mse_loss
            agg_perc_loss += perc_loss
            agg_w_norm_loss += w_norm_loss
            agg_id_loss = id_loss
            agg_loss += loss

        writer.add_scalar('W/ID Loss', agg_id_loss / len(target_indices), step)
        writer.add_scalar('W/MSE Loss', agg_mse_loss / len(target_indices), step)
        writer.add_scalar('W/Perceptual Loss', agg_perc_loss / len(target_indices), step)
        writer.add_scalar('W/Dist to Avg Loss', agg_w_norm_loss / len(target_indices), step)
        writer.add_scalar('W/Combined Loss', agg_loss / len(target_indices), step)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # depth step
        if use_depth_reg:
            agg_depth_loss = 0
            random_cam_index = np.random.choice(len(target_indices))
            random_cam = images[target_indices[random_cam_index]].c_item.c
            for w_index, w_opt in enumerate(w_opt_list):
                image_depth = G.synthesis(w_opt, c=random_cam, noise_mode='const')['image_depth']
                loss = depth_loss_model(view_index=random_cam_index, w_index=w_index, depth_image=image_depth)
                if isinstance(loss, torch.Tensor):
                    loss.backward()
                agg_depth_loss += loss
            depth_loss_model.initialized_list[random_cam_index] = True
            writer.add_scalar('W/Depth Loss', agg_depth_loss / len(target_indices), step)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # interpolation step
        if use_interpolation:
            perc_loss_agg = 0
            mse_loss_agg = 0
            for i in inter_indices:
                target_img = images[i]
                target_cam = target_img.c_item.c
                selected_cams = [images[i].c_item.c for i in target_indices]
                w = interpolate_w_by_cam(w_opt_list, selected_cams, target_cam)
                synth_image = G.synthesis(w, c=target_cam, noise_mode='const')['image']
                perc_loss = perc(target_img.feature, synth_image, vgg=vgg, downsampling=downsampling)
                mse_loss = mse(target_img.target_tensor, synth_image)
                id_loss = id_loss_model(synth_image=synth_image, target_image=images[i].target_tensor)
                loss = 0.1 * mse_loss + perc_loss + id_loss
                loss.backward()

                perc_loss_agg += perc_loss
                mse_loss_agg += mse_loss
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            writer.add_scalar('W/Interpolate MSE', mse_loss_agg / len(inter_indices), step)
            writer.add_scalar('W/Interpolate Perc', perc_loss_agg / len(inter_indices), step)

        # Save projected W for each optimization step.
        for i, w_opt in enumerate(w_opt_list):
            w_out[step][i] = w_opt.detach().cpu()[0]

        # save results
        if step == num_steps - 1 or step % 100 == 0:
            for count, pair in enumerate(zip(target_indices, w_opt_list)):
                i, w_opt = pair
                with torch.no_grad():
                    synth_image = G.synthesis(w_opt, c=images[i].c_item.c, noise_mode='const')['image']
                    synth_image = (synth_image + 1) * (255 / 2)
                    synth_image = synth_image.clamp(0, 255).to(torch.uint8)[0]
                target_image = ((images[i].target_tensor[0] + 1) * (255 / 2)).to(torch.uint8)
                synth_image_comb = torch.concat([target_image, synth_image], dim=-1)
                writer.add_image(f"W/Inversion {i}", synth_image_comb, global_step=step)
                if i == 0:
                    synth_image = synth_image.permute(1, 2, 0).cpu().numpy()
                    PIL.Image.fromarray(synth_image, 'RGB').save(f'{outdir}/{step}.png')

            w_opt_list_np = [w_opt.detach().cpu().numpy() for w_opt in w_opt_list]
            cam_list_np = [images[i].c_item.c.detach().cpu().numpy() for i in target_indices]
            filename = f'{outdir}/{step}_projected_w_mult.npz'
            if step == num_steps - 1:
                filename = f'{outdir}/final_projected_w.npz'
            np.savez(filename, ws=w_opt_list_np, cs=cam_list_np)

    if w_out.shape[2] == 1:
        w_out = w_out.repeat([1, 1, G.mapping.num_ws, 1])

    return w_out
