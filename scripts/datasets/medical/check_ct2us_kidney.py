import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.ct2us_kidney import get_ct2us_kidney_loader


sys.path.append("..")


def check_ct2us_kidney():
    from util import ROOT

    loader = get_ct2us_kidney_loader(
        path=os.path.join(ROOT, "ct2us_kidney"),
        patch_shape=(256, 256),
        batch_size=2,
        download=True,
        resize_inputs=False,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_ct2us_kidney()
