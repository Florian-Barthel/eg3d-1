import subprocess
import os
import argparse


class MultiWHandler:
    def __init__(
            self,
            name,
            continue_folder,
            num_steps=1000,
            num_steps_pti=500,
            outdir="out_multi_w_efhq_v1",
            downsampling=True,
            use_interpolation=True,
            depth_reg=True,
            w_norm_reg=True
    ):
        self.args = []
        self.args.append(f"--name={name}")
        self.args.append(f"--continue-folder={continue_folder}")
        self.args.append(f"--num-steps={num_steps}")
        self.args.append(f"--num-steps-pti={num_steps_pti}")
        self.args.append(f"--outdir={outdir}")
        self.args.append(f"--downsampling={downsampling}")
        self.args.append(f"--use-interpolation={use_interpolation}")
        self.args.append(f"--depth-reg={depth_reg}")
        self.args.append(f"--w-norm-reg={w_norm_reg}")

        self.python = "python"
        self.path_to_program = "multi_inversion_multi_w.py"
        subprocess.run([self.python, self.path_to_program, *self.args], shell=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("gpu", type=int)
    args = parser.parse_args()
    gpu_index = args.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_index}"

    dataset_folder = "./out_efhq_v3"
    all_folders = sorted(os.listdir(dataset_folder))

    for i in range(0, len(all_folders), 4):
        folder = os.path.join(dataset_folder, all_folders[i + gpu_index])
        print(f"Running on {folder}")
        MultiWHandler(name=str(i + gpu_index), continue_folder=folder)
