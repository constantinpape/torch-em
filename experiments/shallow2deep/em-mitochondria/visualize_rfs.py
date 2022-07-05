import argparse
import h5py
from torch_em.shallow2deep import visualize_pretrained_rfs
from torch_em.transform.raw import normalize


def visualize_rfs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", required=True)
    args = parser.parse_args()
    n_forests = 24

    raw_path = "/scratch/pape/vnc/vnc_test.h5"
    with h5py.File(raw_path, "r") as f:
        raw = f["raw"][10]
    raw = normalize(raw)

    visualize_pretrained_rfs(args.checkpoint, raw, n_forests, n_threads=8)


if __name__ == "__main__":
    visualize_rfs()
