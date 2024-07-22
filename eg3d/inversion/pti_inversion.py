import copy
from typing import List
import PIL.Image
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from inversion.load_data import ImageItem
from inversion.utils import create_vgg_features
from inversion.custom_vgg import CustomVGG, NvidiaVGG16
from inversion.loss import mse, perc, IDLoss


def project_pti(
        G,
        images: List[ImageItem],  # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
        w_pivot: torch.Tensor,
        *,
        num_steps=1000,
        initial_learning_rate=3e-4,
        lr_rampdown_length=0.25,
        lr_rampup_length=0.05,
        device: torch.device,
        outdir: str,
        target_indices: List[int],
        writer: SummaryWriter,
        downsampling: bool
):
    assert images[0].target_tensor[0].shape == (G.img_channels, G.img_resolution, G.img_resolution)

    G = copy.deepcopy(G).train().requires_grad_(True).to(device)  # type: ignore
    # vgg = CustomVGG("vgg19").to(device)
    vgg = NvidiaVGG16(device=device)
    create_vgg_features(images, vgg, downsampling)
    id_loss_model = IDLoss()

    w_pivot = w_pivot.to(device).detach()
    optimizer = torch.optim.Adam(G.parameters(), betas=(0.9, 0.999), lr=initial_learning_rate)

    out_params = []

    pbar = tqdm(range(num_steps))
    for step in pbar:
        agg_mse_loss = 0
        agg_perc_loss = 0
        agg_id_loss = 0
        agg_loss = 0

        for i in target_indices:
            synth_images = G.synthesis(w_pivot, c=images[i].c_item.c, noise_mode='const')['image']

            mse_scale = 1 if i == len(target_indices) // 2 else 0
            mse_loss = mse(images[i].target_tensor, synth_images) * mse_scale
            perc_loss = perc(images[i].feature, synth_images, vgg, downsampling=downsampling)#  * mse_scale

            id_loss = id_loss_model(synth_image=synth_images, target_image=images[i].target_tensor)

            loss = 0.1 * mse_loss + perc_loss + id_loss
            loss.backward()

            agg_mse_loss += mse_loss
            agg_perc_loss += perc_loss
            agg_id_loss += id_loss
            agg_loss += loss

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

        if step == num_steps - 1 or step % 100 == 0:
            out_params.append(copy.deepcopy(G).eval().requires_grad_(False).cpu())
            for i in target_indices:
                with torch.no_grad():
                    synth_image = G.synthesis(w_pivot, c=images[i].c_item.c, noise_mode='const')['image']
                    synth_image = (synth_image + 1) * (255 / 2)
                    synth_image = synth_image.clamp(0, 255).to(torch.uint8)[0]
                target_image = ((images[i].target_tensor[0] + 1) * (255 / 2)).to(torch.uint8)
                synth_image_comb = torch.concat([target_image, synth_image], dim=-1)
                writer.add_image(f"PTI/Inversion {i}", synth_image_comb, global_step=step)

                if i == 0:
                    synth_image = synth_image.permute(1, 2, 0).cpu().numpy()
                    PIL.Image.fromarray(synth_image, 'RGB').save(f'{outdir}/PIT_{step}.png')

    return out_params
