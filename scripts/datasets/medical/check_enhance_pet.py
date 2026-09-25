import os

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_enhance_pet_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_enhance_pet():
    loader = get_enhance_pet_loader(
        path=os.path.join(DATA_ROOT, "enhance_pet"),
        batch_size=1,
        patch_shape=(32, 128, 128),
        n_cases=4,
        download=True,
    )
    check_loader(loader, 4, instance_labels=True, plt=True, save_path="test.png")


if __name__ == "__main__":
    check_enhance_pet()
