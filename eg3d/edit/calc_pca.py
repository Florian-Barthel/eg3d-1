import numpy as np
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm

import dnnlib
import click

import legacy
from camera_utils import LookAtPoseSampler


@click.command()
@click.option('--network', help='Network pickle filename', required=True)
@click.option('--n-components', required=False, default=100)
@click.option('--num-samples', required=False, default=1000000)
def run(
        network: str,
        n_components: int,
        num_samples
):
    # Load networks.
    print('Loading networks from "%s"...' % network)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network) as fp:
        network_data = legacy.load_network_pkl(fp)
        G = network_data['G_ema'].requires_grad_(False).to(device)  # type: ignore

    G.rendering_kwargs["ray_start"] = 2.35

    ws = []
    max_samples_per_iter = 10000
    num_iter = num_samples // max_samples_per_iter

    for i in tqdm(range(num_iter)):
        z_samples = np.random.randn(max_samples_per_iter, G.z_dim)
        camera_lookat_point = torch.tensor([0, 0, 0.0], device=device)
        cam2world_pose = LookAtPoseSampler.sample(3.14 / 2, 3.14 / 2, camera_lookat_point, radius=2.7, device=device)
        intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
        c_samples = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        with torch.no_grad():
            w_samples = G.mapping(torch.from_numpy(z_samples).to(device), c_samples.repeat(max_samples_per_iter, 1))  # [N, L, C]
        w_samples = w_samples[:, 0, :].cpu().numpy().astype(np.float32)  # [N, C]
        ws.append(w_samples)

    ws = np.concatenate(ws, axis=0)
    pca = PCA(n_components=n_components)
    pca.fit(ws)
    components = pca.components_
    mean = pca.mean_

    filename = f'edit/w_pca_{n_components}_components_{num_samples}_iterations.npz'
    np.savez(filename, components=components, mean=mean)
    print(components)


if __name__ == "__main__":
    run()
