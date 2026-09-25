import os

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.lumase import get_lumase_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_lumase():
    loader = get_lumase_loader(
        path=os.path.join(DATA_ROOT, "lumase"),
        batch_size=2,
        patch_shape=(1, 256, 256),
        ndim=2,
        resize_inputs=False,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test_lumase.png")


if __name__ == "__main__":
    check_lumase()
