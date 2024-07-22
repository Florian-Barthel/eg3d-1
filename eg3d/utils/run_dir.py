import glob
import re
import os


def get_pkl_and_w(rundir: str, verbose=False):
    network_pkl = rundir + "/fintuned_generator.pkl"
    w_path = rundir + "/final_projected_w.npz"

    if not os.path.exists(w_path):
        w_files = glob.glob(rundir + "/*.npz")
        max_num = -1
        max_file = ""
        for w_file in w_files:
            w_file = w_file.split("/")[-1].split("\\")[-1]
            all_nums = re.findall(r'\d+', w_file)
            if len(all_nums) == 0:
                continue
            current_num = int(all_nums[0])
            if current_num > max_num:
                max_num = current_num
                max_file = w_file
        w_path = rundir + "/" + max_file

    if verbose:
        print("Loading:")
        print(f"\tLatent:    {w_path}")
        print(f"\tGenerator: {network_pkl}")

    return network_pkl, w_path
