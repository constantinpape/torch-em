import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.multi_organ_abdominal_ct import get_multi_organ_abdominal_ct_loader


sys.path.append("..")


def check_multi_organ_abdominal_ct():
    from util import ROOT

    loader = get_multi_organ_abdominal_ct_loader(
        path=os.path.join(ROOT, "multi_organ_abdominal_ct"),
        patch_shape=(1, 512, 512),
        batch_size=1,
        source="tcia",
        ndim=2,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_multi_organ_abdominal_ct()
