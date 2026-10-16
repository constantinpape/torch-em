import os
import sys
import argparse

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.electron_microscopy.mitoverse import get_mitoverse_loader

sys.path.append("..")


def check_mitoverse(path, datasets, benchmark, split, patch_shape, n_samples, save_path):
    from util import ROOT

    loader = get_mitoverse_loader(
        path=os.path.join(ROOT, "mitoverse") if path is None else path,
        batch_size=1,
        patch_shape=patch_shape,
        datasets=datasets,
        benchmark=benchmark,
        split=split,
        download=True,
        sampler=MinInstanceSampler(min_num_instances=2),
    )
    print(f"The loader has {len(loader)} batches.")
    check_loader(loader, n_samples, instance_labels=True, plt=save_path is not None, save_path=save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=None, help="Where to cache the data. Defaults to 'mitoverse' in ROOT.")
    parser.add_argument("--datasets", nargs="+", default=["haberl18"], help="The source datasets, e.g. casser20 han24.")
    parser.add_argument("--all", action="store_true", help="Use (and download) all datasets, about 28 GB.")
    parser.add_argument("--benchmark", default=None, help="Restrict the volumes to a benchmark, e.g. guay21.")
    parser.add_argument("--split", default=None, help="The benchmark split: train, val or test.")
    parser.add_argument("--patch_shape", type=int, nargs=3, default=[8, 128, 128], help="The patch shape.")
    parser.add_argument("-n", "--n_samples", type=int, default=4, help="The number of samples to display.")
    parser.add_argument("--save_path", default=None, help="Save a matplotlib figure here instead of using napari.")
    args = parser.parse_args()
    datasets = None if args.all else args.datasets
    check_mitoverse(
        args.path, datasets, args.benchmark, args.split, tuple(args.patch_shape), args.n_samples, args.save_path
    )


if __name__ == "__main__":
    main()
