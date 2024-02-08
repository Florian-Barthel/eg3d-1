# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import argparse
import batch_mtcnn
from prepare_data import extract_frames
import crop_images_in_the_wild
import face2idr_mat
import preprocess_face_cameras
import shutil


def run_preprocessing_pipeline(src_path, dst_path):
    # from video to pngs
    orig_dst_path = dst_path
    dst_path = os.path.join(dst_path, "frames")
    extract_frames.convert_video(src_path, out_path=dst_path)

    # run mtcnn needed for Deep3DFaceRecon
    batch_mtcnn.run(dst_path, out_path=os.path.join(dst_path, "detections"))

    # run Deep3DFaceRecon
    command = "python Deep3DFaceRecon_pytorch/test.py --img_folder=" + dst_path + " --gpu_ids=0 --name=pretrained --epoch=20"
    print(command)
    os.system(command)

    # crop out the input image
    crop_images_in_the_wild.run(indir=dst_path)

    # convert the pose to our format
    face2idr_mat.run(in_dir=os.path.join(dst_path, "deep_3d_predictions"), out_path=os.path.join(dst_path, 'crop', 'cameras.json'))

    # additional correction to match the submission version
    preprocess_face_cameras.run(source=os.path.join(dst_path, 'crop'), dest=os.path.join(orig_dst_path, "dataset"), mode="orig", max_images=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)

    args = parser.parse_args()
    indir = args.indir.replace("\\", "/")
    outdir = args.outdir.replace("\\", "/")

    for date_folder in os.listdir(indir):
        for id_folder in os.listdir(os.path.join(indir, date_folder)):
            src_path = os.path.join(indir, date_folder, id_folder)
            dst_path = os.path.join(outdir, date_folder + "+" + id_folder)
            os.makedirs(dst_path, exist_ok=True)

            if os.path.exists(os.path.join(dst_path, "dataset", "dataset.json")):
                print("Skipping", dst_path)
                continue

            video_folder = os.path.join(dst_path, "video")
            os.makedirs(video_folder, exist_ok=True)
            for video_file in os.listdir(src_path):
                shutil.copy(os.path.join(src_path, video_file), os.path.join(video_folder, video_file))

            run_preprocessing_pipeline(src_path, dst_path)
