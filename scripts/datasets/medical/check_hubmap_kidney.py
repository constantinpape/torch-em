import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical import get_hubmap_kidney_loader


sys.path.append("..")


def check_hubmap_kidney():
    from util import ROOT

    loader = get_hubmap_kidney_loader(
        path=os.path.join(ROOT, "hubmap_kidney"),
        patch_shape=(512, 512),
        batch_size=2,
        download=True,
        resize_inputs=False,
        sampler=MinInstanceSampler(min_num_instances=1),
    )

    check_loader(loader, 8, plt=True, save_path="./hubmap_kidney.png")


if __name__ == "__main__":
    check_hubmap_kidney()
