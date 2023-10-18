from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_bcss_loader


# set this path to where you have downloaded the bcss data
BCSS_ROOT = "/scratch/projects/nim00007/data/bcss/"


# NOTE: the bcss data cannot be downloaded automatically.
# you need to download it yourself from https://bcsegmentation.grand-challenge.org/BCSS/
def check_bcss():
    loader = get_bcss_loader(
        path=BCSS_ROOT,
        patch_shape=(512, 512),
        batch_size=2,
        label_choices=None,
        download=False
    )
    check_loader(loader, 8, instance_labels=True, rgb=True)


if __name__ == "__main__":
    check_bcss()
