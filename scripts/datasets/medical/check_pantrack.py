import os

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.pantrack import get_pantrack_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_pantrack():
    loader = get_pantrack_loader(
        path=os.path.join(DATA_ROOT, "pantrack"),
        patch_shape=(32, 512, 512),
        batch_size=1,
        sampler=MinInstanceSampler(),
        download=False,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_pantrack()
