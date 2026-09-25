import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.us_sim_and_seg import get_us_sim_and_seg_loader


sys.path.append("..")


def check_us_sim_and_seg():
    from util import ROOT

    loader = get_us_sim_and_seg_loader(
        path=os.path.join(ROOT, "us_sim_and_seg"),
        batch_size=2,
        patch_shape=(512, 512),
        source="aus",
        split="train",
        download=True,
        resize_inputs=True,
    )
    check_loader(loader, 8, plt=True, save_path="./test_aus.png")

    loader = get_us_sim_and_seg_loader(
        path=os.path.join(ROOT, "us_sim_and_seg"),
        batch_size=2,
        patch_shape=(512, 512),
        source="rus",
        split="test",
        download=True,
        resize_inputs=True,
    )
    check_loader(loader, 8, plt=True, save_path="./test_rus.png")


if __name__ == "__main__":
    check_us_sim_and_seg()
