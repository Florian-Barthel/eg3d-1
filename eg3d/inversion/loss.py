import torch
import torch.nn.functional as F


def mse(target_images: torch.tensor, synth_images: torch.tensor):
    return (target_images - synth_images).square().mean()


def perc(
        target_feature: torch.tensor,
        synth_images: torch.tensor,
        vgg: torch.nn.Module,
        downsampling: bool,
):
    # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
    if synth_images.shape[2] > 256 and downsampling:
        synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

    # Features for synth images.
    synth_features = vgg(synth_images)  # , resize_images=False, return_lpips=True)
    return (target_feature.to("cuda") - synth_features).square().sum().mean()


def noise_reg(noise_bufs):
    reg_loss = 0
    for v in noise_bufs.values():
        noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
        while True:
            reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
            reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
            if noise.shape[2] <= 8:
                break
            noise = F.avg_pool2d(noise, kernel_size=2)
    return reg_loss
