import subprocess


class Preprocess:
    def __init__(
            self,
            indir
    ):
        self.args = []
        self.args.append(f"--indir={indir}")
        self.python = "python"
        self.path_to_program = "preprocess_in_the_wild.py"
        subprocess.run([self.python, self.path_to_program, *self.args], shell=False)


if __name__ == "__main__":
    Preprocess(indir="C:/Users/flori/Documents/eg3d-1/dataset_preprocessing/ffhq/Neurohum_all/MF0010_0003/std_neu")
    Preprocess(indir="C:/Users/flori/Documents/eg3d-1/dataset_preprocessing/ffhq/Neurohum_all/MF0010_0004/std_neu")
    Preprocess(indir="C:/Users/flori/Documents/eg3d-1/dataset_preprocessing/ffhq/Neurohum_all/MF0010_0005/std_neu")
    Preprocess(indir="C:/Users/flori/Documents/eg3d-1/dataset_preprocessing/ffhq/Neurohum_all/MF0010_0006/std_neu")
    Preprocess(indir="C:/Users/flori/Documents/eg3d-1/dataset_preprocessing/ffhq/Neurohum_all/MF0010_0007/std_neu")
    Preprocess(indir="C:/Users/flori/Documents/eg3d-1/dataset_preprocessing/ffhq/Neurohum_all/MF0010_0008/std_neu")
    Preprocess(indir="C:/Users/flori/Documents/eg3d-1/dataset_preprocessing/ffhq/Neurohum_all/MF0010_0009/std_neu")
    Preprocess(indir="C:/Users/flori/Documents/eg3d-1/dataset_preprocessing/ffhq/Neurohum_all/MF0010_0010/std_neu")
    Preprocess(indir="C:/Users/flori/Documents/eg3d-1/dataset_preprocessing/ffhq/Neurohum_all/MF0010_0011/std_neu")
    Preprocess(indir="C:/Users/flori/Documents/eg3d-1/dataset_preprocessing/ffhq/Neurohum_all/MF0010_0012/std_neu")
    Preprocess(indir="C:/Users/flori/Documents/eg3d-1/dataset_preprocessing/ffhq/Neurohum_all/MF0010_0013/std_neu")
    Preprocess(indir="C:/Users/flori/Documents/eg3d-1/dataset_preprocessing/ffhq/Neurohum_all/MF0010_0014/std_neu")
