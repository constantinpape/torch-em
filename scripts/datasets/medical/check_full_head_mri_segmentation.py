import os

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.full_head_mri_segmentation import get_full_head_mri_segmentation_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_full_head_mri_segmentation():
    loader = get_full_head_mri_segmentation_loader(
        path=os.path.join(DATA_ROOT, "full_head_mri"),
        patch_shape=(32, 256, 256),
        batch_size=1,
        sampler=MinInstanceSampler(),
        download=False,
    )

    check_loader(loader, 4, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_full_head_mri_segmentation()
