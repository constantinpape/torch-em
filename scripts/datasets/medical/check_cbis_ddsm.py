import os
import sys

from torch_em.data import MinInstanceSampler
from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_cbis_ddsm_loader


sys.path.append("..")


def check_cbis_ddsm():
    from util import ROOT

    loader = get_cbis_ddsm_loader(
        path=os.path.join(ROOT, "cbis_ddsm"),
        patch_shape=(512, 512),
        batch_size=2,
        split="Val",
        task=None,
        tumour_type=None,
        resize_inputs=True,
        sampler=MinInstanceSampler(),
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./cbis_ddsm.png")


if __name__ == "__main__":
    check_cbis_ddsm()
