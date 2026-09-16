import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.advanced_mri_breast_lesions import get_advanced_mri_breast_lesions_loader


sys.path.append("..")


def check_advanced_mri_breast_lesions():
    from util import ROOT

    loader = get_advanced_mri_breast_lesions_loader(
        path=os.path.join(ROOT, "advanced_mri_breast_lesions"),
        patch_shape=(1, 512, 512),
        batch_size=1,
        sampler=MinInstanceSampler(),
        ndim=2,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_advanced_mri_breast_lesions()
