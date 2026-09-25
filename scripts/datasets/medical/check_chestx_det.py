import os

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_chestx_det_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_chestx_det():
    loader = get_chestx_det_loader(
        path=os.path.join(DATA_ROOT, "chestx_det"),
        patch_shape=(512, 512),
        batch_size=1,
        split="train",
        resize_inputs=True,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./chestx_det.png")


if __name__ == "__main__":
    check_chestx_det()
