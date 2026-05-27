import os

import matplotlib
matplotlib.use("Agg")

from torch_em.data.datasets import get_wildenberg_loader  # noqa
from torch_em.util.debug import check_loader  # noqa

CIDAS_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"

# Small sub-region of the p105 annotation extent for quick download/check (~150 z-planes).
P105_BBOX = (576, 24576, 576, 24576, 160, 6160)


def check_wildenberg(label_choice="psd"):
    instance_labels = label_choice == "saturated"
    loader = get_wildenberg_loader(
        os.path.join(CIDAS_ROOT, "wildenberg2023"), batch_size=1, patch_shape=(16, 256, 256),
        download=True, experiments=["p105"], label_choice=label_choice, bounding_box=P105_BBOX,
    )
    check_loader(
        loader, 8, instance_labels=instance_labels, plt=True,
        save_path=f"./check_wildenberg_{label_choice}.png"
    )


def main():
    check_wildenberg("psd")
    check_wildenberg("vesicle")
    check_wildenberg("saturated")


if __name__ == "__main__":
    main()
