import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_deepliif_loader
from torch_em.data.datasets.histopathology.deepliif import TISSUES


sys.path.append("..")


def check_deepliif():
    from util import ROOT

    for split in ["train", "val", "test"]:
        for tissue in TISSUES:
            if tissue == "breast" and split == "test":
                continue

            loader = get_deepliif_loader(
                path=os.path.join(ROOT, "deepliif"),
                batch_size=2,
                patch_shape=(512, 512),
                split=split,
                tissue=tissue,
                modality="ihc",
                label_choice="instances",
                download=True,
                shuffle=True,
            )

            check_loader(loader, 8, instance_labels=True, rgb=True)


if __name__ == "__main__":
    check_deepliif()
