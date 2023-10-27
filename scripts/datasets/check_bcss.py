import numpy as np
from typing import Optional, List

import torch_em
from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_bcss_loader


# set this path to where you have downloaded the bcss data
BCSS_ROOT = "/scratch/projects/nim00007/data/bcss/"


class BCSSLabelTrafo():
    def __init__(
            self,
            label_choices: Optional[List[int]] = None
    ):
        self.label_choices = label_choices

    def __call__(
            self,
            labels: np.ndarray
    ) -> np.ndarray:
        """Returns the transformed labels (use-case for SAM)"""
        if self.label_choices is not None:
            labels[~np.isin(labels, self.label_choices)] = 0
            segmentation = torch_em.transform.label.label_consecutive(labels)
        else:
            segmentation = torch_em.transform.label.label_consecutive(labels)

        return segmentation


# NOTE: the bcss data cannot be downloaded automatically.
# you need to download it yourself from https://bcsegmentation.grand-challenge.org/BCSS/
def check_bcss():
    # loader below checks for selective labels
    chosen_label_loader = get_bcss_loader(
        path=BCSS_ROOT,
        patch_shape=(512, 512),
        batch_size=1,
        download=False,
        label_transform=BCSSLabelTrafo(label_choices=[0, 1, 2])
    )
    print("Length of loader:", len(chosen_label_loader))
    check_loader(chosen_label_loader, 8, instance_labels=True, rgb=True, plt=True, save_path="./bcss.png")


if __name__ == "__main__":
    check_bcss()
