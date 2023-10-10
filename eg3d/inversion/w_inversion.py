import copy
from typing import List
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from tqdm import tqdm

from camera_utils import LookAtPoseSampler
from inversion.utils import get_vgg16, convert_float_images, create_vgg16_features


def project(
        G,
        targets: List[torch.Tensor],  # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
        cs: List[torch.Tensor],
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
        verbose=False,
        device: torch.device,
        outdir: str,
        optimize_cam=True,
        target_indices
):
    assert targets[0].shape == (G.img_channels, G.img_resolution, G.img_resolution)

    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device)  # type: ignore

    # Compute w stats.
    logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    camera_lookat_point = torch.tensor([0, 0, 0.0], device=device)
    cam2world_pose = LookAtPoseSampler.sample(3.14 / 2, 3.14 / 2, camera_lookat_point, radius=2.7, device=device)
    intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
    c_samples = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), c_samples.repeat(w_avg_samples, 1))  # [N, L, C]
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)  # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=True)  # [1, 1, C]
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    # Setup noise inputs.
    noise_bufs = {name: buf for (name, buf) in G.backbone.synthesis.named_buffers() if 'noise_const' in name}

    vgg16 = get_vgg16(device)

    target_images = convert_float_images(targets, device)
    target_features = create_vgg16_features(target_images, vgg16)

    w_avg = torch.tensor(w_avg, dtype=torch.float32, device=device).repeat(1, G.backbone.mapping.num_ws, 1)
    w_opt = w_avg.detach().clone()
    w_opt.requires_grad = True
    w_out = torch.zeros([num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device="cpu")

    original_cs = copy.deepcopy(cs)
    trainable_vars = [w_opt]
    if optimize_noise:
        trainable_vars += list(noise_bufs.values())

    optimizer = torch.optim.Adam(trainable_vars, betas=(0.9, 0.999), lr=initial_learning_rate)

    # optimize camera parameters of input data
    for c in cs:
        c.requires_grad = True
    trainable_vars += cs
    cam_optimizer = torch.optim.Adam(cs, lr=0.0001)

    # Init noise.
    if optimize_noise:
        for buf in noise_bufs.values():
            buf[:] = torch.randn_like(buf)
            buf.requires_grad = True

    for step in range(num_steps):
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

        pbar = tqdm(target_indices)
        for i in pbar:
            # Synth images from opt_w.
            w_noise = torch.randn_like(w_opt) * w_noise_scale
            ws = w_opt + w_noise
            synth_image = G.synthesis(ws, c=cs[i], noise_mode='const')['image']

            # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
            synth_images_perc = (synth_image + 1) * (255 / 2)
            if synth_images_perc.shape[2] > 256:
                synth_images_perc = F.interpolate(synth_images_perc, size=(256, 256), mode='area')

            # Features for synth images.
            synth_features = vgg16(synth_images_perc, resize_images=False, return_lpips=True)
            perc_loss = (target_features[i].to("cuda") - synth_features).square().sum(1).mean()
            mse_loss = (target_images[i] - synth_image).square().mean()
            w_norm_loss = (w_opt - w_avg).square().mean()

            # Noise regularization.
            reg_loss = 0
            if optimize_noise:
                for v in noise_bufs.values():
                    noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
                    while True:
                        reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
                        reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
                        if noise.shape[2] <= 8:
                            break
                        noise = F.avg_pool2d(noise, kernel_size=2)

            agg_mse_loss += mse_loss
            agg_perc_loss += perc_loss
            agg_w_norm_loss += w_norm_loss
            agg_reg_loss += reg_loss

            if optimize_cam_step:
                loss = mse_loss
                loss.backward()
                cam_optimizer.step()
                cam_optimizer.zero_grad(set_to_none=True)
                pbar.set_description(f'step: {step + 1:>4d}/{num_steps} mse: {mse_loss:<4.2f}')
            else:
                loss = 0.1 * mse_loss + perc_loss + 1.0 * w_norm_loss + reg_loss * regularize_noise_weight
                loss.backward()
                description = f'step: {step + 1:>4d}/{num_steps}'
                description += f" mse: {agg_mse_loss / (i + 1):<4.2f}"
                description += f" perc: {agg_perc_loss / (i + 1):<4.2f}"
                description += f" w_norm: {agg_w_norm_loss / (i + 1):<4.2f}"
                pbar.set_description(description)

        if optimize_cam_step:
            print("Camera Change:", torch.sum(torch.abs(torch.cat(original_cs) - torch.cat(cs))).detach().cpu().numpy())
        else:
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

        if step % 10 == 0:
            synth_image = G.synthesis(w_opt, c=cs[0], noise_mode='const')['image']
            synth_image = (synth_image + 1) * (255 / 2)
            synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
            PIL.Image.fromarray(synth_image, 'RGB').save(f'{outdir}/{step}.png')
            np.savez(f'{outdir}/{step}_projected_w.npz', w=w_opt.detach().cpu().numpy())

    if w_out.shape[1] == 1:
        w_out = w_out.repeat([1, G.mapping.num_ws, 1])

    return w_out