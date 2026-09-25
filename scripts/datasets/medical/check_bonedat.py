import os

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical import get_bonedat_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_bonedat():
    loader = get_bonedat_loader(
        path=os.path.join(DATA_ROOT, "bonedat"),
        patch_shape=(1, 512, 512),
        batch_size=1,
        sampler=MinInstanceSampler(),
        ndim=2,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_bonedat()
