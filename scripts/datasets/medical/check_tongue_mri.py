import os

from torch_em.data import MinForegroundSampler
from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_tongue_mri_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_tongue_mri():
    loader = get_tongue_mri_loader(
        path=os.path.join(DATA_ROOT, "tongue_mri"),
        patch_shape=(1, 224, 224),
        batch_size=2,
        site="BMC",
        ndim=2,
        resize_inputs=True,
        download=True,
        sampler=MinForegroundSampler(min_fraction=0.001),
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_tongue_mri()
