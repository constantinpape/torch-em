import os

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_fugc_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_fugc():
    loader = get_fugc_loader(
        path=os.path.join(DATA_ROOT, "fugc"),
        batch_size=1,
        patch_shape=(336, 544),
        split="train",
        download=True,
    )
    check_loader(loader, 4, instance_labels=True, plt=True, save_path="test.png")


if __name__ == "__main__":
    check_fugc()
