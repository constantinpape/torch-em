import os

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.isles2024 import get_isles2024_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_isles2024():
    loader = get_isles2024_loader(
        path=os.path.join(DATA_ROOT, "isles2024"),
        patch_shape=(1, 512, 512),
        batch_size=1,
        label_choice="lesion",
        modality="dwi",
        sampler=MinInstanceSampler(),
        ndim=2,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_isles2024()
