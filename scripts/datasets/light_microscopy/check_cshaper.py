import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_cshaper_loader


sys.path.append("..")


def check_cshaper():
    from util import ROOT

    # NOTE: Download the CShaper zip manually from:
    # https://doi.org/10.6084/m9.figshare.12839315
    # Place the downloaded zip inside the path below. It will be extracted automatically.
    # Requires: pip install nibabel
    loader = get_cshaper_loader(
        path="/mnt/vast-nhr/projects/cidas/cca/data/cshaper",
        batch_size=1,
        patch_shape=(24, 128, 128),
        split="train",
    )

    check_loader(loader, 4, instance_labels=True, plt=True, save_path="cshaper.png")


if __name__ == "__main__":
    check_cshaper()
