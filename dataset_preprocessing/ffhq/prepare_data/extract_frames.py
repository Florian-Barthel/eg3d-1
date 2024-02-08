import os
import cv2


def convert_video(video_path, out_path):
    print("Converting videos to frames")
    os.makedirs(out_path, exist_ok=True)
    for video_file in os.listdir(video_path):
        vidcap = cv2.VideoCapture(os.path.join(video_path, video_file))
        success, image = vidcap.read()
        count = 0
        while success:
            cv2.imwrite(os.path.join(out_path, f"{count:04d}.png"), image)
            success, image = vidcap.read()
            count += 1
