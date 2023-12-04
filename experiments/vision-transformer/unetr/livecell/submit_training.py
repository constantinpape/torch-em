#! /scratch/usr/nimanwai/mambaforge/envs/sam/bin/python

import os
import shutil
import subprocess
from glob import glob
from datetime import datetime


def write_batch_script(out_path, ini_sam=True, source_choice="torch-em", with_affinity=True):
    """
    inputs:
        source_choice:str - [torch_em / monai] source of the unetr model coming from
        ini_sam: bool - initialize torch-em's unetr implementation with sam encoder weights
    """
    cell_types = ["A172", "BT474", "BV2", "Huh7", "MCF7", "SHSY5Y", "SkBr3", "SKOV3"]

    for i, ctype in enumerate(cell_types):
        batch_script = """#!/bin/bash
#SBATCH -t 2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -p grete:shared
#SBATCH -G A100:1
#SBATCH -c 16
#SBATCH -A gzz0001
"""
        if ini_sam:
            batch_script += f"#SBATCH --job-name=unetr-sam-{source_choice}-{ctype}"
        else:
            batch_script += f"#SBATCH --job-name=unetr-{source_choice}-{ctype}"

        env_name = "monai2" if source_choice == "monai" else "sam"

        batch_script += f"""

source ~/.bashrc
mamba activate {env_name}
python livecell_unetr.py --train """

        add_ctype = f"-c {ctype} "
        add_source_choice = f"--source_choice {source_choice} "
        batch_script += add_ctype + add_source_choice

        if ini_sam:
            batch_script += "--do_sam_ini "

        if with_affinity:
            batch_script += "--with_affinities "

        _op = out_path[:-3] + f"_{i}.sh"

        with open(_op, "w") as f:
            f.write(batch_script)


def submit_slurm():
    """Submit python script that needs gpus with given inputs on a slurm node.
    """
    tmp_folder = os.path.expanduser("./gpu_jobs")
    os.makedirs(tmp_folder, exist_ok=True)

    script_name = "unetr_"
    dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    tmp_name = script_name + dt

    batch_script = os.path.join(tmp_folder, f"{tmp_name}.sh")

    write_batch_script(batch_script)

    for my_script in glob(tmp_folder + "/*"):
        cmd = ["sbatch", my_script]
        subprocess.run(cmd)


if __name__ == "__main__":
    try:
        shutil.rmtree("./gpu_jobs")
    except FileNotFoundError:
        pass

    submit_slurm()
