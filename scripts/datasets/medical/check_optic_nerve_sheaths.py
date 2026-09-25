import os

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.optic_nerve_sheaths import get_optic_nerve_sheaths_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_optic_nerve_sheaths():
    loader = get_optic_nerve_sheaths_loader(
        path=os.path.join(DATA_ROOT, "optic_nerve_sheaths"),
        patch_shape=(256, 256),
        batch_size=1,
        sampler=MinInstanceSampler(),
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test_optic_nerve_sheaths.png")


if __name__ == "__main__":
    check_optic_nerve_sheaths()
