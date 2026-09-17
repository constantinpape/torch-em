import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.radgenome_chestct import get_radgenome_chestct_loader


sys.path.append("..")


def check_radgenome_chestct():
    from util import ROOT

    # NOTE: The dataset is not gated on HuggingFace, so no access token is required. Downloading masks for
    # even a handful of cases still requires the full ~10.5 GB validation split mask archive (masks cannot be
    # fetched per case), so 'max_cases' here only limits how many preprocessed CT volumes are downloaded.
    loader = get_radgenome_chestct_loader(
        path=os.path.join(ROOT, "radgenome_chestct"),
        patch_shape=(1, 256, 256),
        batch_size=1,
        split="validation",
        max_cases=5,
        ndim=2,
        sampler=MinInstanceSampler(),
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_radgenome_chestct()
