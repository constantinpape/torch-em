import os

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.bagls_vf import get_bagls_vf_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_bagls_vf():
    loader = get_bagls_vf_loader(
        path=os.path.join(DATA_ROOT, "bagls_vf"),
        patch_shape=(256, 256),
        batch_size=2,
        split="test",
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_bagls_vf()
