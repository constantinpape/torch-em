import os

from torch_em.util.debug import check_loader
from torch_em.data import MinForegroundSampler
from torch_em.data.datasets.medical.mbas import get_mbas_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_mbas():
    loader = get_mbas_loader(
        path=os.path.join(DATA_ROOT, "mbas"),
        patch_shape=(1, 512, 512),
        batch_size=1,
        split="train",
        ndim=2,
        resize_inputs=True,
        sampler=MinForegroundSampler(min_fraction=0.001),
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_mbas()
