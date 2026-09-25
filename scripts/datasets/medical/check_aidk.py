import os

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_aidk_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_aidk():
    loader = get_aidk_loader(
        path=os.path.join(DATA_ROOT, "aidk"),
        patch_shape=(512, 512),
        batch_size=2,
        task="lesion",
        resize_inputs=True,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_aidk()
