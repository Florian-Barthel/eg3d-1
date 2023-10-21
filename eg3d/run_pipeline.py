import subprocess


class MultiWHandler:
    def __init__(
            self,
            network="networks/var3-128.pkl",
            dataset="../dataset_preprocessing/ffhq/1",
            num_steps=500,
            num_steps_pti=500,
            out_dir="out",
            num_targets=5,
            downsampling=True,
            continue_w="out/20231018-1447_multiview_7/499_projected_w.npz",
            optimize_cam=False,
            use_interpolation=False,
            depth_reg=False,

    ):
        self.args = []
        self.args.append(f"--network={network}")
        self.args.append(f"--target={dataset}")
        self.args.append(f"--num-steps={num_steps}")
        self.args.append(f"--num-steps-pti={num_steps_pti}")
        self.args.append(f"--outdir={out_dir}")
        self.args.append(f"--num-targets={num_targets}")
        self.args.append(f"--downsampling={downsampling}")
        self.args.append(f"--continue-w={continue_w}")
        self.args.append(f"--optimize-cam={optimize_cam}")
        self.args.append(f"--use-interpolation={use_interpolation}")
        self.args.append(f"--depth-reg={depth_reg}")
        self.python = "/home/barthel/miniconda3/envs/eg3d_3/bin/python"

        self.path_to_program = "multi_inversion_multi_w.py"

    def run(self):
        subprocess.run([self.python, self.path_to_program, *self.args], shell=False)


class SingleWHandler:
    def __init__(
            self,
            network="networks/var3-128.pkl",
            dataset="../dataset_preprocessing/ffhq/1",
            num_steps=500,
            num_steps_pti=500,
            out_dir="out",
            num_targets=5,
            downsampling=True,
            optimize_cam=False,
    ):
        self.args = []
        self.args.append(f"--network={network}")
        self.args.append(f"--target={dataset}")
        self.args.append(f"--num-steps={num_steps}")
        self.args.append(f"--num-steps-pti={num_steps_pti}")
        self.args.append(f"--outdir={out_dir}")
        self.args.append(f"--num-targets={num_targets}")
        self.args.append(f"--downsampling={downsampling}")
        self.args.append(f"--optimize-cam={optimize_cam}")
        self.python = "/home/barthel/miniconda3/envs/eg3d_3/bin/python"
        self.path_to_program = "multi_inversion.py"

    def run(self):
        subprocess.run([self.python, self.path_to_program, *self.args], shell=False)


if __name__ == "__main__":
    MultiWHandler(continue_w="out/20231020-1943_multiview_7_iter_500_500/499_projected_w.npz", dataset="../dataset_preprocessing/ffhq/1", num_targets=5).run()
    MultiWHandler(continue_w="out/20231020-1943_multiview_7_iter_500_500/499_projected_w.npz", dataset="../dataset_preprocessing/ffhq/1", num_targets=5, use_interpolation=True).run()
    MultiWHandler(continue_w="out/20231020-1943_multiview_7_iter_500_500/499_projected_w.npz", dataset="../dataset_preprocessing/ffhq/1", num_targets=5, use_interpolation=True, depth_reg=True).run()
