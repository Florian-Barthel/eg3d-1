import glob
import click
import imageio
import numpy as np
from tqdm import tqdm

from inversion.load_data import ImageItem


@click.command()
@click.option('--data-path', required=True, metavar='DIR')
def run(
        data_path: str,
):
    images = []
    device = "cuda"
    img_resolution = 512

    for target_fname in glob.glob(data_path + "/*.jpg"):
        images.append(ImageItem(target_fname, device=device, img_resolution=img_resolution))

    video = imageio.get_writer(f'./orig.mp4', mode='I', fps=30, codec='libx264', bitrate='16M')

    for image in tqdm(images + images[::-1]):
        video.append_data(image.t_uint8)
    video.close()


if __name__ == "__main__":
    run()
