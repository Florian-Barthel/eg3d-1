import copy
from typing import List
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from inversion.utils import create_vgg_features, create_w_stats, interpolate_w_by_cam
from inversion.loss import perc, mse, IDLoss, DepthLoss
from inversion.custom_vgg import CustomVGG, NvidiaVGG16
from inversion.load_data import ImageItem


def project(
        G,
        images: List[ImageItem],
        *,
        num_steps=1000,
        w_avg_samples=10000,
        initial_learning_rate=0.1,
        lr_rampdown_length=0.25,
        lr_rampup_length=0.05,
        optimize_noise=False,
        device: torch.device,
        outdir: str,
        optimize_cam=False,
        target_indices: List[int],
        inter_indices: List[int],
        downsampling=True,
        writer: SummaryWriter,
        continue_checkpoint,
        use_interpolation,
        use_depth_reg: bool
):
    assert images[0].target_tensor[0].shape == (G.img_channels, G.img_resolution, G.img_resolution)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device)
    _, w_std = create_w_stats(G, w_avg_samples, device)
    w_checkpoint = np.load(continue_checkpoint)
    w_checkpoint = torch.tensor(w_checkpoint[("w")]).to(device)

    # Setup noise inputs.
    noise_bufs = {name: buf for (name, buf) in G.backbone.synthesis.named_buffers() if 'noise_const' in name}

    vgg = NvidiaVGG16(device)
    # vgg = CustomVGG("vgg19").to(device)
    create_vgg_features(images, vgg, downsampling=downsampling)
    id_loss_model = IDLoss()
    depth_loss_model = DepthLoss(num_targets=len(target_indices))

    # create latent vectors w for each target image
    # w_avg = torch.tensor(w_avg, dtype=torch.float32, device=device).repeat(1, G.backbone.mapping.num_ws, 1)
    w_opt_list = []
    for _ in range(len(target_indices)):
        w_opt = w_checkpoint.detach().clone()
        w_opt.requires_grad = True
        w_opt_list.append(w_opt)

    w_out = torch.zeros([num_steps, len(target_indices)] + list(w_opt_list[0].shape[1:]), dtype=torch.float32,
                        device="cpu")

    trainable_vars = w_opt_list
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
        # w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimize_cam_step = step % 5 == 0 and step > 0 and optimize_cam
        depth_step = step % 5 == 0 and not optimize_cam_step and use_depth_reg

        agg_mse_loss = 0
        agg_perc_loss = 0
        agg_depth_loss = 0
        agg_w_norm_loss = 0
        agg_loss = 0
        agg_cam_loss = 0
        agg_id_loss = 0

        for count, pair in enumerate(zip(target_indices, w_opt_list)):
            i, w_opt = pair

            if depth_step:
                if count > 0:
                    cam = images[target_indices[count - 1]].c_item.c
                    image_depth = G.synthesis(w_opt, c=cam, noise_mode='const')['image_depth'][0]
                    depth_loss = depth_loss_model(count - 1, image_depth)
                    if depth_loss > 0:
                        depth_loss.backward()
                    agg_depth_loss += depth_loss

                if count < len(target_indices) - 2:
                    cam = images[target_indices[count + 1]].c_item.c
                    image_depth = G.synthesis(w_opt, c=cam, noise_mode='const')['image_depth'][0]
                    depth_loss = depth_loss_model(count + 1, image_depth)
                    if depth_loss > 0:
                        depth_loss.backward()
                    agg_depth_loss += depth_loss

            elif optimize_cam_step:
                synth = G.synthesis(w_opt, c=images[i].c_item.c, noise_mode='const')
                synth_image = synth['image']
                image_depth = synth['image_depth'][0]
                depth_loss_model.update(count, image_depth)

                mse_loss = mse(images[i].target_tensor, synth_image)
                mse_loss.backward()
                cam_optimizer.step()
                cam_optimizer.zero_grad(set_to_none=True)
                agg_cam_loss += mse_loss

            else:
                synth = G.synthesis(w_opt, c=images[i].c_item.c, noise_mode='const')
                synth_image = synth['image']
                image_depth = synth['image_depth'][0]
                depth_loss_model.update(count, image_depth)
                perc_loss = perc(images[i].feature, synth_image, vgg=vgg, downsampling=downsampling)
                mse_loss = mse(images[i].target_tensor, synth_image)
                w_norm_loss = mse(w_opt, w_checkpoint)
                id_loss = id_loss_model(synth_image=synth_image, target_image=images[i].target_tensor)

                loss = 0.1 * mse_loss + perc_loss + 1.0 * w_norm_loss + id_loss
                loss.backward()

                agg_mse_loss += mse_loss
                agg_perc_loss += perc_loss
                agg_w_norm_loss += w_norm_loss
                agg_id_loss = id_loss
                agg_loss += loss

        if depth_step:
            writer.add_scalar('W/Depth Loss', agg_depth_loss / len(target_indices), step)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        elif optimize_cam_step:
            pbar.set_description(f'W Inversion Camera Optimisation: {step + 1:>4d}/{num_steps} mse: {agg_cam_loss / len(target_indices):<4.2f}')
            writer.add_scalar('CAM/MSE Loss', agg_cam_loss / len(target_indices), step)
            current_cams = torch.cat([images[i].c_item.c for i in target_indices])
            original_cams = torch.cat([images[i].original_c_item.c for i in target_indices])
            diff = torch.sum(torch.abs(current_cams - original_cams)).detach().cpu().numpy()
            writer.add_scalar('CAM/Absolute Camera Change', diff, step)

        else:
            description = f'W Inversion: {step + 1:>4d}/{num_steps}'
            description += f" mse: {agg_mse_loss / len(target_indices):<4.2f}"
            description += f" perc: {agg_perc_loss / len(target_indices):<4.2f}"
            description += f" w_norm: {agg_w_norm_loss / len(target_indices):<4.2f}"
            pbar.set_description(description)

            writer.add_scalar('W/ID Loss', agg_id_loss / len(target_indices), step)
            writer.add_scalar('W/MSE Loss', agg_mse_loss / len(target_indices), step)
            writer.add_scalar('W/Perceptual Loss', agg_perc_loss / len(target_indices), step)
            writer.add_scalar('W/Dist to Avg Loss', agg_w_norm_loss / len(target_indices), step)
            writer.add_scalar('W/Combined Loss', agg_loss / len(target_indices), step)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

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

        # Normalize noise.
        if optimize_noise:
            with torch.no_grad():
                for buf in noise_bufs.values():
                    buf -= buf.mean()
                    buf *= buf.square().mean().rsqrt()

        if step == num_steps - 1 or step % 25 == 0:
            for count, pair in enumerate(zip(target_indices, w_opt_list)):
                i, w_opt = pair
                with torch.no_grad():
                    synth_image = G.synthesis(w_opt, c=images[i].c_item.c, noise_mode='const')['image']
                    synth_image = (synth_image + 1) * (255 / 2)
                    synth_image = synth_image.clamp(0, 255).to(torch.uint8)[0]
                target_image = ((images[i].target_tensor[0] + 1) * (255 / 2)).to(torch.uint8)
                synth_image_comb = torch.concatenate([target_image, synth_image], dim=-1)
                writer.add_image(f"W/Inversion {i}", synth_image_comb, global_step=step)
                # if step > 0:
                #     writer.add_image(f"Depth/Squared Diff {i}", torch.square(
                #         torch.mean(all_depth_imgs, dim=0) - all_depth_imgs[0][count]).cpu()[None, ...],
                #                      global_step=step)

                if i == 0:
                    synth_image = synth_image.permute(1, 2, 0).cpu().numpy()
                    PIL.Image.fromarray(synth_image, 'RGB').save(f'{outdir}/{step}.png')

            w_opt_list_np = [w_opt.detach().cpu().numpy() for w_opt in w_opt_list]
            cam_list_np = [images[i].c_item.c.detach().cpu().numpy() for i in target_indices]

            np.savez(f'{outdir}/{step}_projected_w_mult.npz', ws=w_opt_list_np, cs=cam_list_np)

    if w_out.shape[2] == 1:
        w_out = w_out.repeat([1, 1, G.mapping.num_ws, 1])

    return w_out
