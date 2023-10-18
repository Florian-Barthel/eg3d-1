from typing import List
import matplotlib as mpl
import matplotlib.pyplot as plt

from inversion.load_data import ImageItem, load

mpl.use('Qt5Agg')


def compare_cam_plot(images: List[ImageItem], save_path):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(0, 0, 0, marker="x", color="black")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_ylim([-3, 3])
    ax.set_xlim([-3, 3])
    ax.set_zlim([-3, 3])

    for i, img in enumerate(images):
        ax.scatter(*img.c_item.xyz(), marker="o", color="blue")
        ax.scatter(*img.original_c_item.xyz(), marker="*", color="red")

    ax.view_init(160, -90)
    plt.savefig(save_path, dpi=300)


def compare_cam_plot_interpolate(images_list: List[ImageItem], interpolated_list: List[ImageItem]):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(0, 0, 0, marker="x", color="black")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_ylim([-3, 3])
    ax.set_xlim([-3, 3])
    ax.set_zlim([-3, 3])

    for img in images_list:
        ax.scatter(*img.c_item.xyz(), marker="o", color="blue")

    for img in interpolated_list:
        ax.scatter(*img.c_item.xyz(), marker="o", color="red")

    ax.view_init(160, -90)
    plt.show()


if __name__ == '__main__':
    images: List[ImageItem] = load("../../dataset_preprocessing/ffhq/1", img_resolution=512, device="cuda")

    compare_cam_plot(images, "../out/test.png")
