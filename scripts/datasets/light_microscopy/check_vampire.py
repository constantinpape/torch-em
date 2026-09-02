import os
import sys
import argparse

from torch_em.data import MinInstanceSampler
from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_vampire_loader

sys.path.append("..")


def check_vampire(save_path=None):
    from util import ROOT

    path = os.path.join(ROOT, "vampire")
    for sample_type in ("wildtype", "lmna_knockout"):
        loader = get_vampire_loader(
            path=path,
            batch_size=1,
            patch_shape=(512, 512),
            sample_type=sample_type,
            sampler=MinInstanceSampler(),
            download=True,
        )
        if save_path is None:
            type_save_path = None
        else:
            base, extension = os.path.splitext(save_path)
            type_save_path = f"{base}_{sample_type}{extension or '.png'}"
        check_loader(loader, 2, instance_labels=True, plt=type_save_path is not None, save_path=type_save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-path", default=None, help="Save non-interactive previews at this path.")
    args = parser.parse_args()
    check_vampire(args.save_path)


if __name__ == "__main__":
    main()
