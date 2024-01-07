import numpy as np
from figures.training_paths import checkpoints_with_inter, checkpoints_inter_depth_reg, checkpoints_with_reg, checkpoints_without_reg

all_distances = []
all_angles = []

for checkpoint in checkpoints_with_inter:
    ws = np.load(checkpoint + "/499_projected_w_mult.npz")['ws']
    if "inter" in checkpoint:
        ws_new = []
        for i in range(len(ws)):
            ws_new.append(ws[i])
            if i < len(ws) - 1:
                ws_new.append(ws[i] * 0.5 + ws[i+1] * 0.5)
        ws = np.stack(ws_new, axis=0)

    ws = ws.reshape(ws.shape[0], -1)
    dists = []
    for i in range(len(ws) - 1):
        dists.append(np.linalg.norm(ws[i] - ws[i+1]))
    mean_dist = np.array(dists).mean()

    angles = []
    for i in range(1, len(ws) - 1):
        v1 = ws[i-1] - ws[i]
        v2 = ws[i+1] - ws[i]
        if np.sum(v1 + v2) < 0.001:
            angles.append(np.pi)
            continue
        dot = np.dot(v1, v2)
        mag_1 = np.sqrt(v1.dot(v1))
        mag_2 = np.sqrt(v2.dot(v2))
        angles.append(np.arccos(dot / (mag_1 * mag_2)))
    mean_angle = np.mean(np.array(angles))

    print("checkpoint", checkpoint)
    print("mean_dist", mean_dist)
    print("mean_angle", mean_angle)

    all_distances.append(mean_dist)
    all_angles.append(mean_angle)

avg_dist = np.array(all_distances).mean()
avg_angle = np.array(all_angles).mean()

print("total avg")
print("avg_dist", avg_dist)
print("avg_angle", avg_angle / (2 * np.pi) * 360)
