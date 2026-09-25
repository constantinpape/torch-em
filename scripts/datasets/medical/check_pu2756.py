import os

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.pu2756 import get_pu2756_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_pu2756():
    loader = get_pu2756_loader(
        path=os.path.join(DATA_ROOT, "pu2756"),
        patch_shape=(512, 512),
        batch_size=1,
        resize_inputs=True,
        sampler=MinInstanceSampler(),
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test_pu2756.png")


if __name__ == "__main__":
    check_pu2756()
