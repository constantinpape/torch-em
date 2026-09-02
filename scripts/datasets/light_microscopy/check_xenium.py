import os
import sys
import argparse

from torch_em.data import MinInstanceSampler
from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_xenium_loader
from torch_em.data.datasets.light_microscopy.xenium import URLS

sys.path.append("..")


def check_xenium(samples=None, save_path=None):
    from util import ROOT

    path = os.path.join(ROOT, "xenium")
    for sample in samples or list(URLS):
        for label_channel in ("nuclei", "cells"):
            loader = get_xenium_loader(
                path=path,
                batch_size=1,
                patch_shape=(512, 512),
                sample=sample,
                label_channel=label_channel,
                raw_channel="dapi",
                sampler=MinInstanceSampler(min_num_instances=4),
                download=True,
            )
            if save_path is None:
                sample_save_path = None
            else:
                base, extension = os.path.splitext(save_path)
                sample_save_path = f"{base}_{sample}_{label_channel}{extension or '.png'}"
            check_loader(
                loader, 2, instance_labels=True, plt=sample_save_path is not None, save_path=sample_save_path
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", nargs="*", default=None, help=f"Samples to check. Choose from {list(URLS)}.")
    parser.add_argument("--save-path", default=None, help="Save non-interactive previews at this path.")
    args = parser.parse_args()
    check_xenium(args.sample, args.save_path)


if __name__ == "__main__":
    main()
