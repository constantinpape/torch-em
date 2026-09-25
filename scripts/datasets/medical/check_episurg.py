import os

from torch_em.util.debug import check_loader
from torch_em.data import MinForegroundSampler
from torch_em.data.datasets.medical import get_episurg_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_episurg():
    loader = get_episurg_loader(
        path=os.path.join(DATA_ROOT, "episurg"),
        patch_shape=(32, 128, 128),
        batch_size=1,
        ndim=3,
        sampler=MinForegroundSampler(min_fraction=0.001),
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_episurg()
