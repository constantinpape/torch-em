import os
import sys

from torch_em.data.datasets import get_hela_cytonuc_loader
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_hela_cytonuc():
    from util import ROOT

    path = os.path.join(ROOT, "hela_cytonuc")
    settings = [
        ("rgb", "nuclei"),
        ("nuclei", "nuclei"),
        ("cytoplasm", "cytoplasm"),
    ]
    for raw_channel, label_choice in settings:
        loader = get_hela_cytonuc_loader(
            path=path,
            batch_size=1,
            patch_shape=(512, 512),
            split="test",
            raw_channel=raw_channel,
            label_choice=label_choice,
            download=True,
        )
        check_loader(loader, 5, instance_labels=True, rgb=raw_channel == "rgb")


if __name__ == "__main__":
    check_hela_cytonuc()
