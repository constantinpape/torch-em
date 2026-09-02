import os

import matplotlib
matplotlib.use("Agg")

from torch_em.data.datasets.electron_microscopy.human_liver_em import get_human_liver_em_loader  # noqa
from torch_em.util.debug import check_loader  # noqa

CIDAS_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"

# Bbox in tissue: 2048x2048 xy, 16 z-slices (tissue spans x=[1195,18890], y=[469,19570]).
CHECK_BBOX = (2219, 4267, 1493, 3541, 290, 298)


def check_human_liver_em(label_choice="mito"):
    loader = get_human_liver_em_loader(
        os.path.join(CIDAS_ROOT, "human_liver_em"), patch_shape=(8, 512, 512), batch_size=1,
        bounding_boxes=[CHECK_BBOX], label_choice=label_choice, download=True,
    )
    check_loader(loader, 8, plt=True, save_path=f"./check_human_liver_em_{label_choice}.png")


def main():
    check_human_liver_em("mito")
    check_human_liver_em("hepatocyte")
    check_human_liver_em("nucleus")


if __name__ == "__main__":
    main()
