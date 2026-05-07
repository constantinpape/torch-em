import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_abus_loader


sys.path.append("..")


def check_abus():
    from util import ROOT

    loader = get_abus_loader(
        path=os.path.join(ROOT, "abus"),
        patch_shape=(512, 512),
        batch_size=2,
        category="benign",
        split="train",
        image_choice="raw",
        resize_inputs=True,
        download=True,
    )

    check_loader(loader, 8, rgb=True, plt=True, save_path="./abus.png")


if __name__ == "__main__":
    check_abus()
