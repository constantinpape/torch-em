import os

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.totalsegmentator_liver_lesions_mr import (
    get_totalsegmentator_liver_lesions_mr_loader
)


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_totalsegmentator_liver_lesions_mr():
    loader = get_totalsegmentator_liver_lesions_mr_loader(
        path=os.path.join(DATA_ROOT, "totalsegmentator_liver_lesions_mr"),
        patch_shape=(1, 256, 256),
        batch_size=1,
        ndim=2,
        download=True,
        sampler=MinInstanceSampler(),
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_totalsegmentator_liver_lesions_mr()
