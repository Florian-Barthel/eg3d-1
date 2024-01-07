import os
from typing import List

from run_metrics import run_metric


def run_multiple(
        run_dir: str,
        run_desc: str,
        data_path: str,
        num_samples: int,
        original_network: str
):
    run_list: List[str] = os.listdir(run_dir)
    matching_runs = [run_dir + "/" + run for run in run_list if run_desc in run]
    results = []
    for run in matching_runs:
        results.append(run_metric(
            data_path=data_path + "/" + run[-1],
            num_samples=num_samples,
            rundir=run,
            original_network=original_network
        ))
    print()


if __name__ == "__main__":
    run_multiple(
        run_dir="out",
        run_desc="multi_w_targets_5_iter_500_500_inter_depth_reg_data",
        num_samples=100, original_network="networks/var3-128.pkl",
        data_path="dataset_preprocessing/ffhq"
    )
