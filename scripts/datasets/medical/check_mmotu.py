import os

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.mmotu import get_mmotu_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_mmotu():
    loader = get_mmotu_loader(
        path=os.path.join(DATA_ROOT, "mmotu"),
        patch_shape=(512, 512),
        batch_size=1,
        resize_inputs=True,
        sampler=MinInstanceSampler(),
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test_mmotu.png")


if __name__ == "__main__":
    check_mmotu()
