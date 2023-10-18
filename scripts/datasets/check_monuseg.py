from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_monuseg_loader


MONUSEG_ROOT = "/scratch/usr/nimanwai/data/monuseg"


def check_monuseg():
    train_loader = get_monuseg_loader(
        path=MONUSEG_ROOT,
        patch_shape=(512, 512),
        batch_size=2,
        split="train",
        download=True
    )
    test_loader = get_monuseg_loader(
        path=MONUSEG_ROOT,
        patch_shape=(512, 512),
        batch_size=1,
        split="test",
        download=True
    )
    check_loader(train_loader, 15, instance_labels=True, rgb=False)
    check_loader(test_loader, 15, instance_labels=True, rgb=False)


if __name__ == "__main__":
    check_monuseg()
