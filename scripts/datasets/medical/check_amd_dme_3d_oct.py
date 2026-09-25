import os

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_amd_dme_3d_oct_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_amd_dme_3d_oct():
    loader = get_amd_dme_3d_oct_loader(
        path=os.path.join(DATA_ROOT, "amd_dme_3d_oct"),
        batch_size=1,
        patch_shape=(32, 512, 512),
        disease="AMD",
        download=True,
    )
    check_loader(loader, 4, instance_labels=True, plt=True, save_path="test.png")


if __name__ == "__main__":
    check_amd_dme_3d_oct()
