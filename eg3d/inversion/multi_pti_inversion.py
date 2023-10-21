import copy
from typing import List
import PIL.Image
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from inversion.load_data import ImageItem
from inversion.utils import create_vgg_features, interpolate_w_by_cam
from inversion.custom_vgg import CustomVGG, NvidiaVGG16
from inversion.loss import mse, perc, IDLoss, DepthLoss


def project_pti(
        G,
        images: List[ImageItem],  # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
        w_pivots: List[torch.Tensor],
        *,
        num_steps=1000,
        initial_learning_rate=3e-4,
        lr_rampdown_length=0.25,
        lr_rampup_length=0.05,
        device: torch.device,
        outdir: str,
        target_indices: List[int],
        inter_indices: List[int],
        writer: SummaryWriter,
        downsampling: bool,
        use_interpolation: bool,
        use_depth_reg: bool
):
    assert images[0].target_tensor[0].shape == (G.img_channels, G.img_resolution, G.img_resolution)

    G = copy.deepcopy(G).train().requires_grad_(True).to(device)  # type: ignore
    # vgg = CustomVGG("vgg19").to(device)
    vgg = NvidiaVGG16(device=device)
    create_vgg_features(images, vgg, downsampling)
    id_loss_model = IDLoss()
    depth_loss_model = DepthLoss(num_targets=len(target_indices))

    w_pivots = [w_pivot.to(device).detach() for w_pivot in w_pivots]
    optimizer = torch.optim.Adam(G.parameters(), betas=(0.9, 0.999), lr=initial_learning_rate)

    out_params = []
    pbar = tqdm(range(num_steps))
    for step in pbar:
        agg_mse_loss = 0
        agg_perc_loss = 0
        agg_loss = 0
        agg_depth_loss = 0
        agg_id_loss = 0

        depth_step = step % 5 == 0 and use_depth_reg

        for count, pair in enumerate(zip(target_indices, w_pivots)):
            i, w_pivot = pair

            if depth_step:
                if count > 0:
                    cam = images[target_indices[count - 1]].c_item.c
                    image_depth = G.synthesis(w_pivot.unsqueeze(0), c=cam, noise_mode='const')['image_depth'][0]
                    depth_loss = depth_loss_model(count - 1, image_depth)
                    if depth_loss > 0:
                        depth_loss.backward()
                    agg_depth_loss += depth_loss

                if count < len(target_indices) - 2:
                    cam = images[target_indices[count + 1]].c_item.c
                    image_depth = G.synthesis(w_pivot.unsqueeze(0), c=cam, noise_mode='const')['image_depth'][0]
                    depth_loss = depth_loss_model(count + 1, image_depth)
                    if depth_loss > 0:
                        depth_loss.backward()
                    agg_depth_loss += depth_loss
            else:
                synth = G.synthesis(w_pivot.unsqueeze(0), c=images[i].c_item.c, noise_mode='const')
                synth_images = synth['image']
                image_depth = synth['image_depth'][0]
                depth_loss_model.update(count, image_depth)
                perc_loss = perc(images[i].feature, synth_images, vgg, downsampling=downsampling)
                mse_loss = mse(images[i].target_tensor, synth_images)
                id_loss = id_loss_model(synth_image=synth_images, target_image=images[i].target_tensor)
                loss = 0.1 * mse_loss + perc_loss + id_loss
                loss.backward()

                agg_mse_loss += mse_loss
                agg_perc_loss += perc_loss
                agg_id_loss = id_loss
                agg_loss += loss

        if depth_step:
            writer.add_scalar('PTI/Depth Loss', agg_depth_loss / len(target_indices), step)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        else:
            writer.add_scalar('PTI/MSE Loss', agg_mse_loss / len(target_indices), step)
            writer.add_scalar('PTI/Perceptual Loss', agg_perc_loss / len(target_indices), step)
            writer.add_scalar('PTI/ID Loss', agg_id_loss / len(target_indices), step)
            writer.add_scalar('PTI/Combined Loss', agg_loss / len(target_indices), step)

            description = f'PTI Inversion: {step + 1:>4d}/{num_steps}'
            description += f" mse: {agg_mse_loss / len(target_indices):<4.2f}"
            description += f" perc: {agg_perc_loss / len(target_indices):<4.2f}"
            pbar.set_description(description)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if use_interpolation:
            perc_loss_agg = 0
            mse_loss_agg = 0
            for i in inter_indices:
                target_img = images[i]
                target_cam = target_img.c_item.c
                selected_cams = [images[i].c_item.c for i in target_indices]
                w = interpolate_w_by_cam(w_pivots, selected_cams, target_cam)
                synth_image = G.synthesis(w.unsqueeze(0), c=target_cam, noise_mode='const')['image']
                perc_loss = perc(target_img.feature, synth_image, vgg=vgg, downsampling=downsampling)
                mse_loss = mse(target_img.target_tensor, synth_image)
                id_loss = id_loss_model(synth_image=synth_image, target_image=images[i].target_tensor)
                loss = 0.1 * mse_loss + perc_loss + id_loss
                loss.backward()

                perc_loss_agg += perc_loss
                mse_loss_agg += mse_loss
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            writer.add_scalar('PTI/Interpolate MSE', mse_loss_agg / len(inter_indices), step)
            writer.add_scalar('PTI/Interpolate Perc', perc_loss_agg / len(inter_indices), step)

        if step == num_steps - 1 or step % 25 == 0:
            for i, w_pivot in zip(target_indices, w_pivots):
                out_params.append(copy.deepcopy(G).eval().requires_grad_(False).cpu())
                with torch.no_grad():
                    synth_image = G.synthesis(w_pivot.unsqueeze(0), c=images[i].c_item.c, noise_mode='const')['image']
                    synth_image = (synth_image + 1) * (255 / 2)
                    synth_image = synth_image.clamp(0, 255).to(torch.uint8)[0]
                target_image = ((images[i].target_tensor[0] + 1) * (255 / 2)).to(torch.uint8)
                synth_image_comb = torch.concatenate([target_image, synth_image], dim=-1)
                writer.add_image(f"PTI/Inversion {i}", synth_image_comb, global_step=step)

                if i == 0:
                    synth_image = synth_image.permute(1, 2, 0).cpu().numpy()
                    PIL.Image.fromarray(synth_image, 'RGB').save(f'{outdir}/PIT_{step}.png')

    return out_params
