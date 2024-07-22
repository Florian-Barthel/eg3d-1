import numpy as np
from training_paths import single_view, multi_view_9, checkpoints_inter_depth_reg, SPI, checkpoints_without_reg
import matplotlib.pyplot as plt

selected_checkpoints = [single_view, multi_view_9, checkpoints_without_reg, checkpoints_inter_depth_reg]

# selected_checkpoints = [single_view, SPI, multi_view_9, checkpoints_without_reg, checkpoints_inter_depth_reg]
legend = [
    "single-view",
    #"single-view SPI",
    "multi-view",
    "multi-latent",
    "multi-latent + consistency reg."
]

for i, selected_checkpoint in enumerate(selected_checkpoints):
    angles = np.load(selected_checkpoint[0] + "/angles.npz")["values"]
    print(len(angles))
    sort_indices = np.argsort(angles)
    angles = angles[sort_indices]
    angles = angles / (2 * np.pi) * 360 - 90

    angles, unique_indices = np.unique(angles, return_index=True)

    lpips_list = []
    id_list = []
    for checkpoint in selected_checkpoint:
        lpips_list.append(np.load(checkpoint + "/PTI_net_lpips.npz")["values"][sort_indices][unique_indices])
        id_list.append(np.load(checkpoint + "/PTI_net_id_sim.npz")["values"][sort_indices][unique_indices])
    print()

    id_list = np.mean(id_list, axis=0)
    plt.plot(angles, id_list, label=legend[i])

plt.ylabel("ID Similarity")
plt.xlabel("Viewing angle in degree")
plt.legend()
plt.savefig("angle_graph_single_multi.png", dpi=300)
plt.show()
