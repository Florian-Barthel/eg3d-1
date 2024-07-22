import subprocess
import os
import argparse


class SingleWHandler:
    def __init__(
            self,
            dataset,
            num_targets,
            network="networks/ffhqrebalanced512-128.pkl",
            num_steps=500,
            num_steps_pti=500,
            outdir="out",
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
    SingleWHandler(
        dataset="C:/Users/flori/projects/eg3d-1/dataset_preprocessing/ffhq/3DIL_dataset_filter_conf/03_05_17+114-0433",
        num_targets=7,
        num_steps=1000,
        num_steps_pti=1000,
        outdir="out_efhq/ffhqrebalanced512-128",
        optimize_cam=True,
        network="networks/ffhqrebalanced512-128.pkl"
    )

    SingleWHandler(
        dataset="C:/Users/flori/projects/eg3d-1/dataset_preprocessing/ffhq/3DIL_dataset_filter_conf/03_05_17+114-0433",
        num_targets=7,
        num_steps=1000,
        num_steps_pti=1000,
        outdir="out_efhq/var3-128",
        optimize_cam=True,
        network="networks/var3-128.pkl"
    )

    SingleWHandler(
        dataset="C:/Users/flori/projects/eg3d-1/dataset_preprocessing/ffhq/3DIL_dataset_filter_conf/03_05_17+114-0433",
        num_targets=7,
        num_steps=2000,
        num_steps_pti=1000,
        outdir="out_efhq/ffhqrebalanced512-128_2000",
        optimize_cam=True,
        network="networks/ffhqrebalanced512-128.pkl"
    )


    SingleWHandler(
        dataset="C:/Users/flori/projects/eg3d-1/dataset_preprocessing/ffhq/3DIL_dataset_filter_conf/03_05_17+114-0433",
        num_targets=9,
        num_steps=2000,
        num_steps_pti=1000,
        outdir="out_efhq/ffhqrebalanced512-128_2000_9",
        optimize_cam=True,
        network="networks/ffhqrebalanced512-128.pkl"
    )
