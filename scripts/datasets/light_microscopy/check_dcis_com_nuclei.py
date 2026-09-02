import os
import sys

from torch_em.data.datasets import get_dcis_com_nuclei_loader
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_dcis_com_nuclei():
    from util import ROOT

    path = os.path.join(ROOT, "dcis_com_nuclei")
    for split, n_samples in (("train", 5), ("test", 2)):
        loader = get_dcis_com_nuclei_loader(
            path=path,
            batch_size=1,
            patch_shape=(512, 512),
            split=split,
            download=True,
        )
        check_loader(loader, n_samples, instance_labels=True)


if __name__ == "__main__":
    check_dcis_com_nuclei()
