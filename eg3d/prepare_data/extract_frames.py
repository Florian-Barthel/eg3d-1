import os

import cv2


video_path = "../../data/rotations"
out_path = "../../data/rotations_frames"
os.makedirs(out_path, exist_ok=True)

for video_file in os.listdir(video_path):
    frame_path = out_path + "/" + video_file.split(".")[0]
    os.makedirs(frame_path, exist_ok=True)
    vidcap = cv2.VideoCapture(video_path + "/" + video_file)
    success, image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(frame_path + "/frame%d.jpg" % count, image)
        success, image = vidcap.read()
        print("Read a new frame: ", success)
        count += 1
