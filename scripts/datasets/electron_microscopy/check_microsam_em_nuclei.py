import os
import sys
import argparse

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.electron_microscopy.microsam_em_nuclei import get_microsam_em_nuclei_loader

sys.path.append("..")


def check_microsam_em_nuclei(path, source, patch_shape, n_samples, save_path):
    from util import ROOT

    loader = get_microsam_em_nuclei_loader(
        path=os.path.join(ROOT, "microsam_em_nuclei") if path is None else path,
        batch_size=1,
        patch_shape=patch_shape,
        source=source,
        download=True,
        sampler=MinInstanceSampler(),
    )
    print(f"The loader has {len(loader)} batches.")
    check_loader(loader, n_samples, instance_labels=True, plt=save_path is not None, save_path=save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=None, help="Where to cache the data. Defaults to 'microsam_em_nuclei'.")
    sources = ["train", "ilastik_carving", "default_sam", "finetuned_sam"]
    parser.add_argument("--source", default="train", choices=sources, help="The data to use.")
    parser.add_argument("--patch_shape", type=int, nargs=3, default=[32, 256, 256], help="The patch shape.")
    parser.add_argument("-n", "--n_samples", type=int, default=4, help="The number of samples to display.")
    parser.add_argument("--save_path", default=None, help="Save a matplotlib figure here instead of using napari.")
    args = parser.parse_args()
    check_microsam_em_nuclei(args.path, args.source, tuple(args.patch_shape), args.n_samples, args.save_path)


if __name__ == "__main__":
    main()
