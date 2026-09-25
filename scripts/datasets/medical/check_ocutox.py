import os

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.ocutox import get_ocutox_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_ocutox():
    loader = get_ocutox_loader(
        path=os.path.join(DATA_ROOT, "ocutox"),
        patch_shape=(512, 512),
        batch_size=1,
        resize_inputs=True,
        sampler=MinInstanceSampler(),
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test_ocutox.png")


if __name__ == "__main__":
    check_ocutox()
