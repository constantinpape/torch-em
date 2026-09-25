import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_spatialproteomics_bnhl_loader


sys.path.append("..")


def check_spatialproteomics_bnhl(channel):
    from util import ROOT

    loader = get_spatialproteomics_bnhl_loader(
        path=os.path.join(ROOT, "spatialproteomics_bnhl"),
        patch_shape=(256, 256),
        batch_size=1,
        channel=channel,
        download=True,
    )

    check_loader(
        loader, 8, instance_labels=True, rgb=False,
        plt=True, save_path=f"check_spatialproteomics_bnhl_{channel.replace('/', '-')}.png"
    )


if __name__ == "__main__":
    check_spatialproteomics_bnhl(channel="DAPI")
    check_spatialproteomics_bnhl(channel="CD20")
