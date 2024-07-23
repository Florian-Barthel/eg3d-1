import subprocess
import os
import argparse


class SingleWHandler:
    def __init__(
            self,
            dataset,
            num_targets=9,
            network="networks/efhq.pkl",
            num_steps=1000,
            num_steps_pti=500,
            outdir="out_efhq_v3",
            downsampling=True,
            optimize_cam=True,
    ):
        self.args = []
        self.args.append(f"--network={network}")
        self.args.append(f"--target={dataset}")
        self.args.append(f"--num-steps={num_steps}")
        self.args.append(f"--num-steps-pti={num_steps_pti}")
        self.args.append(f"--outdir={outdir}")
        self.args.append(f"--num-targets={num_targets}")
        self.args.append(f"--downsampling={downsampling}")
        self.args.append(f"--optimize-cam={optimize_cam}")
        self.python = "python"
        self.path_to_program = "multi_inversion.py"
        subprocess.run([self.python, self.path_to_program, *self.args], shell=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("gpu", type=int)
    args = parser.parse_args()
    gpu_index = args.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_index}"

    dataset_folder = "../dataset_preprocessing/ffhq/3DIL_dataset_filter_conf"
    all_folders = sorted(os.listdir(dataset_folder))

    for i in range(0, len(all_folders), 4):
        folder = os.path.join(dataset_folder, all_folders[i + gpu_index])
        print(f"Running on {folder}")
        SingleWHandler(dataset=folder)
