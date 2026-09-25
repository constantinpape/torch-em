import argparse
import os

from elf.io import open_file
from torch_em.shallow2deep import visualize_pretrained_rfs
from torch_em.transform.raw import normalize


ROOT = "/scratch/pape/s2d-mitochondria"


def visualize_rfs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", default="mitoem")
    parser.add_argument("-n", "--rf_name", default="rfs2d")
    args = parser.parse_args()

    dataset = args.dataset
    assert dataset in ("mitoem", "vnc")
    if dataset == "mitoem":
        raw_path = os.path.join(ROOT, dataset, "human_test.n5")
    else:
        raw_path = os.path.join(ROOT, dataset, "vnc_test.n5")
    assert os.path.exists(raw_path), raw_path

    rf_folder = os.path.join(ROOT, args.rf_name, dataset)
    assert os.path.exists(rf_folder), rf_folder

    with open_file(raw_path, "r") as f:
        raw = f["raw"][0, :1024, :1024]
    raw = normalize(raw)

    n_forests = 24
    visualize_pretrained_rfs(rf_folder, raw, n_forests, n_threads=8)


if __name__ == "__main__":
    visualize_rfs()
