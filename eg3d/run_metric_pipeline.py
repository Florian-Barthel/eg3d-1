import subprocess


class MetricHandler:
    def __init__(
            self,
            rundir,
            original_network="networks/var3-128.pkl",
            dataset="../dataset_preprocessing/ffhq/1",
            num_samples=200,
    ):
        self.args = []
        self.args.append(f"--original-network={original_network}")
        self.args.append(f"--data-path={dataset}")
        self.args.append(f"--num-samples={num_samples}")
        self.args.append(f"--rundir={rundir}")
        self.python = "/home/barthel/miniconda3/envs/eg3d_3/bin/python"

        self.path_to_program = "run_metrics.py"

    def run(self):
        subprocess.run([self.python, self.path_to_program, *self.args], shell=False)


if __name__ == "__main__":
    MetricHandler(rundir="out/20231020-0914_multiview_1_iter_1000_1000").run()
    MetricHandler(rundir="out/20231020-1110_multiview_3_iter_1000_1000").run()
    MetricHandler(rundir="out/20231020-1138_multiview_5_iter_1000_1000").run()
    MetricHandler(rundir="out/20231020-1224_multiview_7_iter_1000_1000").run()
