import os

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_slice2_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_slice2():
    loader = get_slice2_loader(
        path=os.path.join(DATA_ROOT, "slice2"),
        batch_size=1,
        patch_shape=(64, 128, 128),
        datasets=["DS0011", "DS0004"],
        download=True,
    )
    check_loader(loader, 4, instance_labels=True, plt=True, save_path="test.png")


if __name__ == "__main__":
    check_slice2()
