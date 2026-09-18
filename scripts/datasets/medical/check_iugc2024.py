import os

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.iugc2024 import get_iugc2024_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_iugc2024(split):
    loader = get_iugc2024_loader(
        path=os.path.join(DATA_ROOT, "iugc2024"),
        patch_shape=(512, 512),
        batch_size=2,
        split=split,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path=f"./test_{split}.png")


if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        check_iugc2024(split)
