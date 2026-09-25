import os

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_hvdropdb_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_hvdropdb():
    loader = get_hvdropdb_loader(
        path=os.path.join(DATA_ROOT, "hvdropdb"),
        patch_shape=(512, 512),
        batch_size=2,
        structure="vessels",
        device="RetCam",
        resize_inputs=True,
        download=True,
    )

    check_loader(loader, 8, rgb=True, plt=True, save_path="./hvdropdb.png")


if __name__ == "__main__":
    check_hvdropdb()
