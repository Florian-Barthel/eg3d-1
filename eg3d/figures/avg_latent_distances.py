import numpy as np


checkpoints = []
all_distances = []
all_angles = []

for checkpoint in checkpoints:
    ws = np.load(checkpoint)['ws']
    dists = []
    for i in range(len(ws) - 1):
        dists.append(np.sum(np.square(ws[i] - ws[i+1])))
    mean_dist = np.array(dists).mean()

    angles = []
    for i in range(1, len(ws) - 1):
        v1 = ws[i-1] - ws[i]
        v2 = ws[i+1] - ws[i]
        dot = np.dot(v1, v2)
        mag_1 = np.sqrt(v1.dot(v1))
        mag_2 = np.sqrt(v2.dot(v2))
        angles.append(np.sum(np.square(ws[i] - ws[i+1])))
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
print("avg_angle", avg_angle)
