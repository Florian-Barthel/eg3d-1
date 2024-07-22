# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


import argparse
import cv2
import os
from mtcnn import MTCNN
import random
from tqdm import tqdm
detector = MTCNN()


def run(in_root, out_path):
    os.makedirs(out_path, exist_ok=True)

    imgs = sorted([x for x in os.listdir(in_root) if x.endswith(".jpg") or x.endswith(".png")])
    # random.shuffle(imgs)
    for img in tqdm(imgs, desc="MTCNN detection"):
        src = os.path.join(in_root, img)
        if img.endswith(".jpg"):
            dst = os.path.join(out_path, img.replace(".jpg", ".txt"))
        elif img.endswith(".png"):
            dst = os.path.join(out_path, img.replace(".png", ".txt"))
        else:
            raise NotImplementedError("Image not in [png, jpg]")

        if not os.path.exists(dst):
            image = cv2.cvtColor(cv2.imread(src), cv2.COLOR_BGR2RGB)
            result = detector.detect_faces(image)

            if len(result) > 0:
                index = 0
                if len(result) > 1: # if multiple faces, take the biggest face
                    size = -100000
                    for r in range(len(result)):
                        size_ = result[r]["box"][2] + result[r]["box"][3]
                        if size < size_:
                            size = size_
                            index = r

                keypoints = result[index]['keypoints']
                # print(result[index]["confidence"])
                if result[index]["confidence"] >= 0.99:
                    outLand = open(dst, "w")
                    outLand.write(str(float(keypoints['left_eye'][0])) + " " + str(float(keypoints['left_eye'][1])) + "\n")
                    outLand.write(str(float(keypoints['right_eye'][0])) + " " + str(float(keypoints['right_eye'][1])) + "\n")
                    outLand.write(str(float(keypoints['nose'][0])) + " " + str(float(keypoints['nose'][1])) + "\n")
                    outLand.write(str(float(keypoints['mouth_left'][0])) + " " + str(float(keypoints['mouth_left'][1])) + "\n")
                    outLand.write(str(float(keypoints['mouth_right'][0])) + " " + str(float(keypoints['mouth_right'][1])) + "\n")
                    outLand.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_root', type=str, default="", help='process folder')
    args = parser.parse_args()
    out_path = os.path.join(args.in_root, "detections")
    run(args.in_root, out_path)