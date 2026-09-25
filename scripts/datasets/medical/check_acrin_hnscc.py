import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.acrin_hnscc import get_acrin_hnscc_loader


sys.path.append("..")


def check_acrin_hnscc():
    from util import ROOT

    loader = get_acrin_hnscc_loader(
        path=os.path.join(ROOT, "acrin_hnscc"),
        patch_shape=(1, 256, 256),
        batch_size=1,
        modality="PET",
        ndim=2,
        resize_inputs=True,
        sampler=MinInstanceSampler(),
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_acrin_hnscc()
