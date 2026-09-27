import os

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_air_leish_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_air_leish():
    loader = get_air_leish_loader(
        path=os.path.join(DATA_ROOT, "air_leish"),
        batch_size=2,
        patch_shape=(512, 512),
        target="nuclei",
        download=True,
    )
    check_loader(loader, 4, instance_labels=True, plt=True, save_path="test.png")


if __name__ == "__main__":
    check_air_leish()
