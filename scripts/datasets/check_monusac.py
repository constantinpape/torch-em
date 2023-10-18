from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_monusac_loader


MONUSAC_ROOT = "/scratch/usr/nimanwai/data/monusac"


def check_monusac():
    train_loader = get_monusac_loader(
        path=MONUSAC_ROOT,
        download=True,
        patch_shape=(512, 512),
        batch_size=2,
        split="train"
    )

    test_loader = get_monusac_loader(
        path=MONUSAC_ROOT,
        download=True,
        patch_shape=(512, 512),
        batch_size=1,
        split="test"
    )

    check_loader(train_loader, 8, instance_labels=True, rgb=True)
    check_loader(test_loader, 8, instance_labels=True, rgb=True)


if __name__ == "__main__":
    check_monusac()
