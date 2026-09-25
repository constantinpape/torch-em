import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.soft_tissue_sarcoma import get_soft_tissue_sarcoma_loader


sys.path.append("..")


def check_soft_tissue_sarcoma():
    from util import ROOT

    for modality in ["T2FS", "T1", "PET", "CT"]:
        loader = get_soft_tissue_sarcoma_loader(
            path=os.path.join(ROOT, "soft_tissue_sarcoma"),
            patch_shape=(1, 256, 256),
            batch_size=1,
            modality=modality,
            ndim=2,
            resize_inputs=True,
            sampler=MinInstanceSampler(),
            download=True,
        )

        check_loader(loader, 8, plt=True, save_path=f"./test_{modality}.png")


if __name__ == "__main__":
    check_soft_tissue_sarcoma()
