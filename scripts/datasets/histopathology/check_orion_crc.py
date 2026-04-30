import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_orion_crc_loader


sys.path.append("..")


def check_orion_crc():
    from util import ROOT

    loader = get_orion_crc_loader(
        path=os.path.join(ROOT, "orion_crc"),
        split="train",
        modality="he",
        label_type="instances",
        patch_shape=(512, 512),
        batch_size=2,
        download=False,
    )

    check_loader(loader, 8, instance_labels=True, rgb=True)


if __name__ == "__main__":
    check_orion_crc()
