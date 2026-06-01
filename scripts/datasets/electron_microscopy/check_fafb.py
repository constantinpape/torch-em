import os

import matplotlib
matplotlib.use("Agg")

from torch_em.data.datasets.electron_microscopy.fafb import get_fafb_loader  # noqa
from torch_em.data.sampler import MinInstanceSampler  # noqa
from torch_em.util.debug import check_loader  # noqa

CIDAS_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_fafb(label_choice="neurons"):
    loader = get_fafb_loader(
        os.path.join(CIDAS_ROOT, "fafb"), patch_shape=(256, 640, 640), batch_size=1,
        label_choice=label_choice, download=True,
        sampler=MinInstanceSampler(min_num_instances=2),
    )
    check_loader(loader, 8, instance_labels=True, plt=True, save_path=f"./check_fafb_{label_choice}.png")


def main():
    check_fafb("neurons")
    check_fafb("nuclei")


if __name__ == "__main__":
    main()
