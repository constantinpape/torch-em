import os

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.segrap2025 import get_segrap2025_lnctv_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_segrap2025_lnctv():
    loader = get_segrap2025_lnctv_loader(
        path=os.path.join(DATA_ROOT, "segrap2025_lnctv_full"),
        patch_shape=(32, 512, 512),
        batch_size=1,
        split="train",
        modality="ct",
        sampler=MinInstanceSampler(),
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_segrap2025_lnctv()
