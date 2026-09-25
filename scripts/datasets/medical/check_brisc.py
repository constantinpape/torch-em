import os

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.brisc import get_brisc_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_brisc():
    loader = get_brisc_loader(
        path=os.path.join(DATA_ROOT, "brisc"),
        patch_shape=(512, 512),
        batch_size=1,
        split="test",
        sampler=MinInstanceSampler(),
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_brisc()
