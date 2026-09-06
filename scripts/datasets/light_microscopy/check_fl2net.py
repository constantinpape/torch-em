import os
import sys
import argparse

from torch_em.data.datasets import get_fl2net_loader
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_fl2net(splits, save_path=None):
    from util import ROOT

    # NOTE: Google Drive limits how often it serves the archives. Download 'raw.tar.gz' and
    # 'gt.tar.gz' manually from https://github.com/funalab/FL2-Net and place them in the path below
    # when the automatic download fails.
    path = os.path.join(ROOT, "fl2net")
    for split in splits:
        # NOTE: Each split reads through both archives once, so checking all three splits takes a
        # while. The extracted timepoints are cached, so a second run is fast.
        loader = get_fl2net_loader(
            path=path,
            batch_size=1,
            patch_shape=(32, 96, 96),
            split=split,
            embryos=None,
            timepoints=[300, 400, 500],
            download=True,
        )
        if save_path is None:
            split_save_path = None
        else:
            base, extension = os.path.splitext(save_path)
            split_save_path = f"{base}_{split}{extension or '.png'}"
        check_loader(
            loader, 3, instance_labels=True, plt=split_save_path is not None, save_path=split_save_path
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits", nargs="+", default=["train"], choices=["train", "val", "test"],
        help="The splits to check. Each split reads through both archives once.",
    )
    parser.add_argument("--save-path", default=None, help="Save non-interactive split previews at this path.")
    args = parser.parse_args()
    check_fl2net(args.splits, args.save_path)


if __name__ == "__main__":
    main()
