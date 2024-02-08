import subprocess
import os
import argparse

class SingleWHandler:
    def __init__(
            self,
            dataset,
            num_targets,
            network="networks/var3-128.pkl",
            num_steps=500,
            num_steps_pti=500,
            outdir="out",
            downsampling=True,
            optimize_cam=False,
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


class MultiWHandler:
    def __init__(
            self,
            dataset,
            continue_w,
            num_targets,
            use_interpolation,
            w_norm_reg,
            depth_reg,
            network="networks/var3-128.pkl",
            num_steps=500,
            num_steps_pti=500,
            outdir="out",
            downsampling=True,
    ):
        self.args = []
        self.args.append(f"--network={network}")
        self.args.append(f"--target={dataset}")
        self.args.append(f"--num-steps={num_steps}")
        self.args.append(f"--num-steps-pti={num_steps_pti}")
        self.args.append(f"--outdir={outdir}")
        self.args.append(f"--num-targets={num_targets}")
        self.args.append(f"--downsampling={downsampling}")
        self.args.append(f"--continue-w={continue_w}")
        self.args.append(f"--use-interpolation={use_interpolation}")
        self.args.append(f"--depth-reg={depth_reg}")
        self.args.append(f"--w-norm-reg={w_norm_reg}")
        self.python = "python"
        self.path_to_program = "multi_inversion_multi_w.py"
        subprocess.run([self.python, self.path_to_program, *self.args], shell=False)


if __name__ == "__main__":
    # SingleWHandler(
    #     dataset="../dataset_preprocessing/ffhq/3DIL_dataset/01_06_17+114-0700",
    #     num_targets=7,
    #     num_steps=2,
    #     num_steps_pti=2
    # )

    # MultiWHandler(
    #     dataset="../dataset_preprocessing/ffhq/3DIL_dataset/01_06_17+114-0700",
    #     continue_w="out/20240206-1354_multiview_7_iter_2_2_data_01_06_17+114-0700/1_projected_w.npz",
    #     num_targets=9,
    #     use_interpolation=True,
    #     w_norm_reg=True,
    #     depth_reg=False,
    #     num_steps=10,
    #     num_steps_pti=10
    # )

    parser = argparse.ArgumentParser()
    parser.add_argument("gpu", type=int)
    args = parser.parse_args()
    gpu_index = args.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_index}"

    dataset_folder = "../dataset_preprocessing/ffhq/3DIL_dataset"
    all_folders = sorted(os.listdir(dataset_folder))

    # for i in range(0, len(all_folders), 4):
    #     folder = os.path.join(dataset_folder, all_folders[i + gpu_index])
    #     print(f"Running on {folder}")
    #     SingleWHandler(
    #         dataset=folder,
    #         num_targets=7,
    #         num_steps=500,
    #         num_steps_pti=500,
    #         outdir="out_single_w_3DIL"
    #     )

    checkpoint_folder = "./out_single_w_3DIL"
    for i in range(0, len(all_folders), 4):
        folder = os.path.join(dataset_folder, all_folders[i + gpu_index])
        checkpoint = f"./out_single_w_3DIL/single-w_{all_folders[i + gpu_index]}/499_projected_w.npz"
        print(f"Running on {folder}")
        print(f"Using checkpoint {checkpoint}")
        MultiWHandler(
            dataset=folder,
            continue_w=checkpoint,
            num_targets=5,
            use_interpolation=True,
            w_norm_reg=True,
            depth_reg=True,
            num_steps=500,
            num_steps_pti=500,
            outdir="out_multi_w_3DIL"
        )
