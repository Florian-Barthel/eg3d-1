import copy
from typing import List
import PIL.Image
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from inversion.load_data import ImageItem
from inversion.utils import create_vgg_features, interpolate_w_by_cam
from inversion.custom_vgg import CustomVGG, NvidiaVGG16
from inversion.loss import mse, perc, IDLoss, DepthLossAll


def project_pti(
        G,
        images: List[ImageItem],
        w_pivots: List[torch.Tensor],
        *,
        num_steps=1000,
        initial_learning_rate=3e-4,
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
    G = copy.deepcopy(G).train().requires_grad_(True).to(device)
    vgg = NvidiaVGG16(device=device)
    create_vgg_features(images, vgg, downsampling)
    id_loss_model = IDLoss()
    depth_loss_model = DepthLossAll(num_targets=len(target_indices))
    w_pivots = [w_pivot.to(device).detach() for w_pivot in w_pivots]
    optimizer = torch.optim.Adam(G.parameters(), betas=(0.9, 0.999), lr=initial_learning_rate)

    out_params = []
    for step in tqdm(range(num_steps)):

        # normal step
        agg_mse_loss = 0
        agg_perc_loss = 0
        agg_loss = 0
        agg_id_loss = 0
        for count, pair in enumerate(zip(target_indices, w_pivots)):
            i, w_pivot = pair
            synth = G.synthesis(w_pivot.unsqueeze(0), c=images[i].c_item.c, noise_mode='const')
            synth_images = synth['image']
            perc_loss = perc(images[i].feature, synth_images, vgg, downsampling=downsampling)
            mse_loss = mse(images[i].target_tensor, synth_images)
            id_loss = id_loss_model(synth_image=synth_images, target_image=images[i].target_tensor)
            loss = 0.1 * mse_loss + perc_loss + id_loss
            loss.backward()
            agg_mse_loss += mse_loss
            agg_perc_loss += perc_loss
            agg_id_loss = id_loss
            agg_loss += loss

        writer.add_scalar('PTI/MSE Loss', agg_mse_loss / len(target_indices), step)
        writer.add_scalar('PTI/Perceptual Loss', agg_perc_loss / len(target_indices), step)
        writer.add_scalar('PTI/ID Loss', agg_id_loss / len(target_indices), step)
        writer.add_scalar('PTI/Combined Loss', agg_loss / len(target_indices), step)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # depth step
        if step % 1 == 0 and use_depth_reg:
            agg_depth_loss = 0
            random_cam_index = np.random.choice(len(target_indices))
            random_cam = images[target_indices[random_cam_index]].c_item.c
            for w_index, w_pivot in enumerate(w_pivots):
                image_depth = G.synthesis(w_pivot.unsqueeze(0), c=random_cam, noise_mode='const')['image_depth']
                loss = depth_loss_model(view_index=random_cam_index, w_index=w_index, depth_image=image_depth)
                if isinstance(loss, torch.Tensor):
                    loss.backward()
                agg_depth_loss += loss
            depth_loss_model.initialized_list[random_cam_index] = True
            writer.add_scalar('PTI/Depth Loss', agg_depth_loss / len(target_indices), step)
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

        # save results
        if step == num_steps - 1 or step % 25 == 0:
            for i, w_pivot in zip(target_indices, w_pivots):
                out_params.append(copy.deepcopy(G).eval().requires_grad_(False).cpu())
                with torch.no_grad():
                    synth_image = G.synthesis(w_pivot.unsqueeze(0), c=images[i].c_item.c, noise_mode='const')['image']
                    synth_image = (synth_image + 1) * (255 / 2)
                    synth_image = synth_image.clamp(0, 255).to(torch.uint8)[0]
                target_image = ((images[i].target_tensor[0] + 1) * (255 / 2)).to(torch.uint8)
                synth_image_comb = torch.concat([target_image, synth_image], dim=-1)
                writer.add_image(f"PTI/Inversion {i}", synth_image_comb, global_step=step)
                if i == 0:
                    synth_image = synth_image.permute(1, 2, 0).cpu().numpy()
                    PIL.Image.fromarray(synth_image, 'RGB').save(f'{outdir}/PIT_{step}.png')

    return out_params
