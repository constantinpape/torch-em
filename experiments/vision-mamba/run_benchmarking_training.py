import os
import shutil
import subprocess
from glob import glob
from datetime import datetime


SCRIPT_NAMES = {
    "livecell": "livecell/run_livecell.py",
    "cremi": "cremi/run_cremi.py",
    "neurips-cellseg": "neurips-cellseg/run_neurips_cellseg.py",
    "lm": "light_microscopy/run_lm.py"
}
ALL_MODELS = ["vim_t", "vim_s", "vim_b"]
ALL_SETTINGS = ["boundaries", "affinities", "distances"]


def write_batch_script(out_path, dataset, model_type, mode, setting):
    batch_script = f"""#!/bin/bash
#SBATCH -t 2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -p grete:shared
#SBATCH -G A100:1
#SBATCH -c 16
#SBATCH -A gzz0001
#SBATCH -x ggpu136
#SBATCH --constraint=80gb
#SBATCH --job-name=vimunet-{dataset}

source ~/.bashrc
mamba activate vm3
"""

    python_script = f"python {SCRIPT_NAMES[dataset]} "
    python_script += f"--{mode} "  # train or predict
    python_script += f"-m {model_type} "

    if dataset != "lm":
        python_script += f"--{setting} "  # boundaries / affinities / distances

    batch_script += python_script

    _op = out_path[:-3] + f"{model_type}-{setting}-{dataset}.sh"

    with open(_op, "w") as f:
        f.write(batch_script)


def write_all_scripts(batch_script, mode, dataset=None, model_type=None, setting=None):
    if dataset is None:
        dataset = list(SCRIPT_NAMES.keys())
    else:
        dataset = [dataset]

    if model_type is None:
        model_type = ALL_MODELS
    else:
        model_type = [model_type]

    if setting is None:
        setting = ALL_SETTINGS
    else:
        setting = [setting]

    for _dataset in dataset:
        for _model in model_type:
            # we overwrite setting for the lm training
            setting = ["distances"] if _dataset == "lm" else setting
            for _setting in setting:
                write_batch_script(
                    out_path=batch_script,
                    dataset=_dataset,
                    model_type=_model,
                    mode=mode,
                    setting=_setting
                )


def submit_slurm(args):
    """Submit python script that needs gpus with given inputs on a slurm node.
    """
    tmp_folder = os.path.expanduser("./gpu_jobs")
    os.makedirs(tmp_folder, exist_ok=True)

    script_name = "vimunet_"
    dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    tmp_name = script_name + dt

    batch_script = os.path.join(tmp_folder, f"{tmp_name}.sh")

    write_all_scripts(
        batch_script, mode=args.phase, dataset=args.dataset, model_type=args.model_type, setting=args.setting
    )

    for my_script in glob(tmp_folder + "/*"):
        cmd = ["sbatch", my_script]
        subprocess.run(cmd)


if __name__ == "__main__":
    try:
        shutil.rmtree("./gpu_jobs")
    except FileNotFoundError:
        pass

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default=None)
    parser.add_argument("-m", "--model_type", type=str, default=None)
    parser.add_argument("-p", "--phase", type=str, default=None, required=True)
    parser.add_argument("--setting", type=str, default=None)
    args = parser.parse_args()
    submit_slurm(args)
