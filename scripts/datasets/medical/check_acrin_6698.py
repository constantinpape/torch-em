import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.acrin_6698 import get_acrin_6698_loader


sys.path.append("..")


def check_acrin_6698():
    from util import ROOT

    # NOTE: The collection has 1110 relevant VOLSER masks, so 'max_cases' is used here to only
    # download and preprocess a small subset.
    loader = get_acrin_6698_loader(
        path=os.path.join(ROOT, "acrin_6698"),
        patch_shape=(1, 256, 256),
        batch_size=1,
        max_cases=5,
        ndim=2,
        sampler=MinInstanceSampler(),
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_acrin_6698()
