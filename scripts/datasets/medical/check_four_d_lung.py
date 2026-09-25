import os

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_four_d_lung_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_four_d_lung():
    loader = get_four_d_lung_loader(
        path=os.path.join(DATA_ROOT, "four_d_lung"),
        batch_size=1,
        patch_shape=(32, 512, 512),
        download=True,
    )
    check_loader(loader, 4, instance_labels=True, plt=True, save_path="test.png")


if __name__ == "__main__":
    check_four_d_lung()
