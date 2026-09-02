import os
import argparse

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_histo_miner_loader


DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "histo_miner")


def check_histo_miner(path, task, save_path):
    loader = get_histo_miner_loader(
        path=path,
        batch_size=1,
        patch_shape=(512, 512),
        split="train",
        task=task,
        resize_inputs=(task == "tumor"),
        download=True,
        shuffle=True,
    )
    check_loader(
        loader, 4, instance_labels=(task == "nuclei"), plt=save_path is not None, rgb=True, save_path=save_path
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=DEFAULT_ROOT, help="Folder for the Histo-Miner data.")
    parser.add_argument("--task", default="nuclei", choices=["nuclei", "tumor"], help="The task to check.")
    parser.add_argument("--save-path", default=None, help="Optional path for a non-interactive loader check.")
    args = parser.parse_args()
    check_histo_miner(args.path, args.task, args.save_path)


if __name__ == "__main__":
    main()
