import os
import sys

from torch_em.data.datasets.electron_microscopy import get_emcopilot_loader
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_emcopilot(name="Au_ZSM5"):
    from util import ROOT

    loader = get_emcopilot_loader(
        path=os.path.join(ROOT, "emcopilot"),
        patch_shape=(512, 512),
        batch_size=2,
        name=name,
        download=True,
        shuffle=True,
    )
    check_loader(loader, 8, instance_labels=False, plt=True, save_path="./emcopilot.png")


if __name__ == "__main__":
    check_emcopilot()
