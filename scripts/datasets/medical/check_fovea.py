import os

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_fovea_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_fovea():
    loader = get_fovea_loader(
        path=os.path.join(DATA_ROOT, "fovea"),
        batch_size=1,
        patch_shape=(512, 512),
        domain="both",
        annotation="vessels",
        annotator=1,
        ndim=2,
        download=True,
    )
    check_loader(loader, 8, plt=True, save_path="./test_fovea.png")


if __name__ == "__main__":
    check_fovea()
