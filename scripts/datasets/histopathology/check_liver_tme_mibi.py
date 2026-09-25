import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets.histopathology.liver_tme_mibi import get_liver_tme_mibi_loader


sys.path.append("..")

# Restrict to a couple of FOVs for a quick check: the full dataset is about 140 GB.
FOVS = [
    "pCSL005_CSL021_Hep-PK_MIBI_slide1-D1_0_0",
    "pCSL005_CSL021_Hep-PK_MIBI_slide1-D1_0_1",
]


def check_liver_tme_mibi(channel):
    from util import ROOT

    loader = get_liver_tme_mibi_loader(
        path=os.path.join(ROOT, "liver_tme_mibi"),
        patch_shape=(256, 256),
        batch_size=1,
        fovs=FOVS,
        channel=channel,
        download=True,
    )

    check_loader(
        loader, 8, instance_labels=True, rgb=False,
        plt=True, save_path=f"check_liver_tme_mibi_{channel}.png"
    )


if __name__ == "__main__":
    check_liver_tme_mibi(channel="PanCK")
    check_liver_tme_mibi(channel="HH3")
