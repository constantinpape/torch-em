import os

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_liver_hcc_seg_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_liver_hcc_seg():
    loader = get_liver_hcc_seg_loader(
        path=os.path.join(DATA_ROOT, "liver_hcc_seg"),
        batch_size=1,
        patch_shape=(8, 256, 256),
        target="tumor",
        rater=1,
        download=True,
    )
    check_loader(loader, 4, instance_labels=True, plt=True, save_path="test.png")


if __name__ == "__main__":
    check_liver_hcc_seg()
