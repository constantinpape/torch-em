import os

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_denpar_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_denpar():
    loader = get_denpar_loader(
        path=os.path.join(DATA_ROOT, "denpar"),
        patch_shape=(512, 512),
        batch_size=2,
        split="train",
        label_choice="instance",
        resize_inputs=True,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./denpar.png")


if __name__ == "__main__":
    check_denpar()
