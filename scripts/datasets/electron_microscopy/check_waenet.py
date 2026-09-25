import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_waenet_loader


sys.path.append("..")


def check_waenet():
    from util import ROOT

    loader = get_waenet_loader(
        path=os.path.join(ROOT, "waenet"),
        dataset_id=1,
        patch_shape=(512, 512),
        batch_size=1,
        label_type="nucleus",
        split="train",
        download=True,
    )
    check_loader(loader, 4, instance_labels=False, plt=True, save_path="./waenet.png")


if __name__ == "__main__":
    check_waenet()
