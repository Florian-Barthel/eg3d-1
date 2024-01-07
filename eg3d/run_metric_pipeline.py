import subprocess


class MetricHandler:
    def __init__(
            self,
            rundir,
            original_network="networks/var3-128.pkl",
            dataset="../dataset_preprocessing/ffhq/1",
            num_samples=180,
            run_w_plus=False
    ):
        print("------------------------------------------------------------------")
        print("------------------------------------------------------------------")
        print(dataset)
        self.args = []
        self.args.append(f"--original-network={original_network}")
        self.args.append(f"--data-path={dataset}")
        self.args.append(f"--num-samples={num_samples}")
        self.args.append(f"--rundir={rundir}")
        self.args.append(f"--run-w-plus={run_w_plus}")
        self.python = "python"

        self.path_to_program = "run_metrics.py"
        subprocess.run([self.python, self.path_to_program, *self.args], shell=False)


if __name__ == "__main__":
    samples_list = [1, 180]
    for samples in samples_list:
        MetricHandler(rundir="out/20231018-1531_multiview_9", dataset="../dataset_preprocessing/ffhq/1", num_samples=samples)
        MetricHandler(rundir="out/20231025-0059_multiview_9_iter_500_500_data_2", dataset="../dataset_preprocessing/ffhq/2", num_samples=samples)
        MetricHandler(rundir="out/20231025-0141_multiview_9_iter_500_500_data_3", dataset="../dataset_preprocessing/ffhq/3", num_samples=samples)
        MetricHandler(rundir="out/20231025-0224_multiview_9_iter_500_500_data_4", dataset="../dataset_preprocessing/ffhq/4", num_samples=samples)
        MetricHandler(rundir="out/20231025-0306_multiview_9_iter_500_500_data_5", dataset="../dataset_preprocessing/ffhq/5", num_samples=samples)
        MetricHandler(rundir="out/20231025-0349_multiview_9_iter_500_500_data_6", dataset="../dataset_preprocessing/ffhq/6", num_samples=samples)



