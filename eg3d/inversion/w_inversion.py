import copy
from typing import List
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


from inversion.utils import create_vgg_features, create_w_stats
from inversion.loss import perc, mse, noise_reg, IDLoss
from inversion.plots import compare_cam_plot
from inversion.custom_vgg import CustomVGG, NvidiaVGG16
from inversion.load_data import ImageItem


def project(
        G,
        images: List[ImageItem],
        *,
        num_steps=1000,
        w_avg_samples=10000,
        initial_learning_rate=0.1,
        initial_noise_factor=0.05,
        lr_rampdown_length=0.25,
        lr_rampup_length=0.05,
        noise_ramp_length=0.75,
        regularize_noise_weight=1e5,
        optimize_noise=False,
        device: torch.device,
        outdir: str,
        optimize_cam=False,
        target_indices,
        downsampling=True,
        writer: SummaryWriter,
        w_plus: bool = True
):
    assert images[0].target_tensor[0].shape == (G.img_channels, G.img_resolution, G.img_resolution)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device)
    w_avg, w_std = create_w_stats(G, w_avg_samples, device)

    # Setup noise inputs.
    noise_bufs = {name: buf for (name, buf) in G.backbone.synthesis.named_buffers() if 'noise_const' in name}

    vgg = NvidiaVGG16(device)
    # vgg = CustomVGG("vgg19").to(device)
    create_vgg_features(images, vgg, downsampling=downsampling)
    id_loss_model = IDLoss()

    w_avg = torch.tensor(w_avg, dtype=torch.float32, device=device).repeat(1, G.backbone.mapping.num_ws, 1)
    w_opt = w_avg.detach().clone()
    w_opt.requires_grad = True
    w_out = torch.zeros([num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device="cpu")

    trainable_vars = [w_opt]
    if optimize_noise:
        trainable_vars += list(noise_bufs.values())

    optimizer = torch.optim.Adam(trainable_vars, betas=(0.9, 0.999), lr=initial_learning_rate)

    # optimize camera parameters of input data
    cam_parameters = []
    for img in images:
        img.c_item.c.requires_grad = True
        cam_parameters.append(img.c_item.c)
    cam_optimizer = torch.optim.Adam(cam_parameters, lr=0.0001)

    # Init noise.
    if optimize_noise:
        for buf in noise_bufs.values():
            buf[:] = torch.randn_like(buf)
            buf.requires_grad = True

    pbar = tqdm(range(num_steps))
    for step in pbar:
        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimize_cam_step = step % 5 == 0 and step > 0 and optimize_cam

        agg_mse_loss = 0
        agg_perc_loss = 0
        agg_w_norm_loss = 0
        agg_reg_loss = 0
        agg_id_loss = 0
        agg_loss = 0

        for i in target_indices:
            # Synth images from opt_w.
            w_noise = torch.randn_like(w_opt) * w_noise_scale
            ws = w_opt + w_noise
            synth_image = G.synthesis(ws, c=images[i].c_item.c, noise_mode='const')['image']

            perc_loss = perc(images[i].feature, synth_image, vgg=vgg, downsampling=downsampling)
            mse_loss = mse(images[i].target_tensor, synth_image)
            w_norm_loss = mse(w_opt, w_avg)
            id_loss = id_loss_model(synth_image=synth_image, target_image=images[i].target_tensor)

            # Noise regularization.
            reg_loss = 0
            if optimize_noise:
                reg_loss = noise_reg(noise_bufs)

            if optimize_cam_step:
                loss = mse_loss
                loss.backward()
                cam_optimizer.step()
                cam_optimizer.zero_grad(set_to_none=True)
            else:
                loss = 0.1 * mse_loss + perc_loss + 1.0 * w_norm_loss + reg_loss * regularize_noise_weight + id_loss
                loss.backward()

            agg_mse_loss += mse_loss
            agg_perc_loss += perc_loss
            agg_w_norm_loss += w_norm_loss
            agg_reg_loss += reg_loss
            agg_id_loss = id_loss
            agg_loss += loss

        if optimize_cam_step:
            pbar.set_description(f'W Inversion Camera Optimisation: {step + 1:>4d}/{num_steps} mse: {agg_mse_loss / len(target_indices):<4.2f}')
            writer.add_scalar('CAM/MSE Loss', agg_mse_loss/ len(target_indices), step)
            current_cams = torch.cat([images[i].c_item.c for i in target_indices])
            original_cams = torch.cat([images[i].original_c_item.c for i in target_indices])
            writer.add_scalar('CAM/Absolute Camera Change', torch.sum(torch.abs(current_cams - original_cams)).detach().cpu().numpy(), step)
            compare_cam_plot([images[i] for i in target_indices], save_path=outdir + f"/{step}_cam_plot.png")
        else:
            description = f'W Inversion: {step + 1:>4d}/{num_steps}'
            description += f" mse: {agg_mse_loss / len(target_indices):<4.2f}"
            description += f" perc: {agg_perc_loss / len(target_indices):<4.2f}"
            description += f" w_norm: {agg_w_norm_loss / len(target_indices):<4.2f}"
            pbar.set_description(description)

            writer.add_scalar('W/MSE Loss', agg_mse_loss / len(target_indices), step)
            writer.add_scalar('W/Perceptual Loss', agg_perc_loss / len(target_indices), step)
            writer.add_scalar('W/Dist to Avg Loss', agg_w_norm_loss / len(target_indices), step)
            writer.add_scalar('W/ID Loss', agg_id_loss / len(target_indices), step)
            writer.add_scalar('W/Combined Loss', agg_loss / len(target_indices), step)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # Save projected W for each optimization step.
        w_out[step] = w_opt.detach().cpu()[0]

        # Normalize noise.
        if optimize_noise:
            with torch.no_grad():
                for buf in noise_bufs.values():
                    buf -= buf.mean()
                    buf *= buf.square().mean().rsqrt()

        if step % 25 == 0 or step == num_steps - 1:
            for i in target_indices:
                with torch.no_grad():
                    synth_image = G.synthesis(w_opt, c=images[i].c_item.c, noise_mode='const')['image']
                    synth_image = (synth_image + 1) * (255 / 2)
                    synth_image = synth_image.clamp(0, 255).to(torch.uint8)[0]
                target_image = ((images[i].target_tensor[0] + 1) * (255 / 2)).to(torch.uint8)
                synth_image_comb = torch.concatenate([target_image, synth_image], dim=-1)
                writer.add_image(f"W/Inversion {i}", synth_image_comb, global_step=step)

                if i == 0:
                    synth_image = synth_image.permute(1, 2, 0).cpu().numpy()
                    PIL.Image.fromarray(synth_image, 'RGB').save(f'{outdir}/{step}.png')
            np.savez(f'{outdir}/{step}_projected_w.npz', w=w_opt.detach().cpu().numpy())

    if w_out.shape[1] == 1:
        w_out = w_out.repeat([1, G.mapping.num_ws, 1])

    return w_out
