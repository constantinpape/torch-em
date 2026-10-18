import os
import sys
import argparse

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.electron_microscopy.vem_nuclei import get_vem_nuclei_loader

sys.path.append("..")


def check_vem_nuclei(path, dimensionality, patch_shape, n_samples, save_path):
    from util import ROOT

    loader = get_vem_nuclei_loader(
        path=os.path.join(ROOT, "vem_nuclei") if path is None else path,
        batch_size=1,
        patch_shape=patch_shape,
        dimensionality=dimensionality,
        download=True,
        sampler=MinInstanceSampler(),
    )
    print(f"The loader has {len(loader)} batches.")
    check_loader(loader, n_samples, instance_labels=True, plt=save_path is not None, save_path=save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=None, help="Where to cache the data. Defaults to 'vem_nuclei' in ROOT.")
    parser.add_argument("--dimensionality", default="2d", choices=["2d", "3d"], help="The 2D or 3D data.")
    parser.add_argument("--patch_shape", type=int, nargs="+", default=None, help="The patch shape.")
    parser.add_argument("-n", "--n_samples", type=int, default=4, help="The number of samples to display.")
    parser.add_argument("--save_path", default=None, help="Save a matplotlib figure here instead of using napari.")
    args = parser.parse_args()
    patch_shape = args.patch_shape or ([512, 512] if args.dimensionality == "2d" else [32, 256, 256])
    check_vem_nuclei(args.path, args.dimensionality, tuple(patch_shape), args.n_samples, args.save_path)


if __name__ == "__main__":
    main()
