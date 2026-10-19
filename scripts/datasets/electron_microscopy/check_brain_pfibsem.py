import os
import sys

from torch_em.data.datasets.electron_microscopy import get_brain_pfibsem_loader
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_brain_pfibsem(segmentation_type="mitochondria", input_type="reference"):
    from util import ROOT

    loader = get_brain_pfibsem_loader(
        path=os.path.join(ROOT, "brain_pfibsem"),
        patch_shape=(32, 256, 256),
        batch_size=1,
        segmentation_type=segmentation_type,
        input_type=input_type,
        split="train",
        download=True,
        shuffle=True,
    )
    check_loader(loader, 4, instance_labels=True, plt=True, save_path=f"./brain_pfibsem_{segmentation_type}.png")


if __name__ == "__main__":
    check_brain_pfibsem()
