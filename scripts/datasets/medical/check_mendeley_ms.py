import os
import sys

from torch_em.data import MinInstanceSampler
from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.mendeley_ms import get_mendeley_ms_loader


sys.path.append("..")


def check_mendeley_ms():
    from util import ROOT

    loader = get_mendeley_ms_loader(
        path=os.path.join(ROOT, "mendeley_ms"),
        patch_shape=(1, 256, 256),
        batch_size=1,
        modality="flair",
        ndim=2,
        download=True,
        sampler=MinInstanceSampler(min_num_instances=2),
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_mendeley_ms()
