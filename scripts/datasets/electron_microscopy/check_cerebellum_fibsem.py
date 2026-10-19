import os
import sys

from torch_em.data.datasets.electron_microscopy import get_cerebellum_fibsem_loader
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_cerebellum_fibsem(segmentation_type="mitochondria"):
    from util import ROOT

    loader = get_cerebellum_fibsem_loader(
        path=os.path.join(ROOT, "cerebellum_fibsem"),
        patch_shape=(32, 256, 256),
        batch_size=1,
        segmentation_type=segmentation_type,
        split="train",
        download=True,
        shuffle=True,
    )
    check_loader(loader, 4, instance_labels=True, plt=True, save_path=f"./cerebellum_fibsem_{segmentation_type}.png")


if __name__ == "__main__":
    check_cerebellum_fibsem()
