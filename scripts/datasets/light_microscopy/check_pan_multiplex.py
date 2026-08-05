import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_pan_multiplex_loader


sys.path.append("..")


def check_pan_multiplex():
    from util import ROOT

    for raw_channel in ["stacked", "nuclei"]:
        loader = get_pan_multiplex_loader(
            path=os.path.join(ROOT, "pan_multiplex"),
            patch_shape=(512, 512),
            batch_size=2,
            subset="mibi_decidua",
            split="train",
            raw_channel=raw_channel,
            download=True,
            shuffle=True,
        )

        check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_pan_multiplex()
