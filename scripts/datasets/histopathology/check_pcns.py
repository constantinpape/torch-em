import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_pcns_loader


sys.path.append("..")


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data/pcns"


def check_pcns(cancer_type=None):
    loader = get_pcns_loader(
        path=DATA_ROOT,
        batch_size=1,
        patch_shape=(256, 256),
        split="train",
        cancer_type=cancer_type,
        download=False,
    )

    check_loader(loader, 8, instance_labels=True, rgb=True, plt=True, save_path="check_pcns.png")


if __name__ == "__main__":
    check_pcns()
