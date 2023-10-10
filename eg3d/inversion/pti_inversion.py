
import copy
from typing import List
import PIL.Image
import torch
import torch.nn.functional as F
from tqdm import tqdm
from inversion.utils import convert_float_images, create_vgg16_features, get_vgg16


def project_pti(
        G,
        targets: List[torch.Tensor],  # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
        w_pivot: torch.Tensor,
        cs: List[torch.Tensor],
        *,
        num_steps=1000,
        initial_learning_rate=3e-4,
        lr_rampdown_length=0.25,
        lr_rampup_length=0.05,
        device: torch.device,
        outdir: str,
        target_indices
):
    assert targets[0].shape == (G.img_channels, G.img_resolution, G.img_resolution)

    G = copy.deepcopy(G).train().requires_grad_(True).to(device)  # type: ignore
    vgg16 = get_vgg16(device)
    target_images = convert_float_images(targets, device)
    target_features = create_vgg16_features(target_images, vgg16)

    w_pivot = w_pivot.to(device).detach()
    optimizer = torch.optim.Adam(G.parameters(), betas=(0.9, 0.999), lr=initial_learning_rate)

    out_params = []

    for step in range(num_steps):
        pbar = tqdm(target_indices)
        agg_mse_loss = 0
        agg_perc_loss = 0

        for i in pbar:
            synth_images = G.synthesis(w_pivot, c=cs[i], noise_mode='const')['image']

            # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
            synth_images_perc = (synth_images + 1) * (255 / 2)
            if synth_images_perc.shape[2] > 256:
                synth_images_perc = F.interpolate(synth_images_perc, size=(256, 256), mode='area')

            # Features for synth images.
            synth_features = vgg16(synth_images_perc, resize_images=False, return_lpips=True)
            perc_loss = (target_features[i].to("cuda") - synth_features).square().sum(1).mean()
            mse_loss = (target_images[i] - synth_images).square().mean()
            loss = 0.1 * mse_loss + perc_loss
            loss.backward()

            agg_mse_loss += mse_loss
            agg_perc_loss += perc_loss

            description = f'step: {step + 1:>4d}/{num_steps}'
            description += f" mse: {agg_mse_loss / (i + 1):<4.2f}"
            description += f" perc: {agg_perc_loss / (i + 1):<4.2f}"
            pbar.set_description(description)


        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step == num_steps - 1 or step % 10 == 0:
            out_params.append(copy.deepcopy(G).eval().requires_grad_(False).cpu())
            synth_image = G.synthesis(w_pivot, c=cs[0], noise_mode='const')['image']
            synth_image = (synth_image + 1) * (255 / 2)
            synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
            PIL.Image.fromarray(synth_image, 'RGB').save(f'{outdir}/PIT_{step}.png')

    return out_params

