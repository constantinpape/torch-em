import os

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_rim_one_dl_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_rim_one_dl():
    loader = get_rim_one_dl_loader(
        path=os.path.join(DATA_ROOT, "rim_one_dl"),
        patch_shape=(512, 512),
        batch_size=2,
        split="train",
        resize_inputs=True,
        download=True,
    )

    check_loader(loader, 8, rgb=True, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_rim_one_dl()
