import os

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_ut_endomri_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_ut_endomri():
    loader = get_ut_endomri_loader(
        path=os.path.join(DATA_ROOT, "ut_endomri"),
        batch_size=1,
        patch_shape=(8, 256, 256),
        dataset="D1",
        sequence="T2",
        structure="ut",
        rater=1,
        download=True,
    )
    check_loader(loader, 4, plt=True, save_path="test.png")


if __name__ == "__main__":
    check_ut_endomri()
