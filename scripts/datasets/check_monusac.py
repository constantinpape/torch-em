from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_monusac_loader


MONUSAC_ROOT = "/scratch/usr/nimanwai/data/monusac"


def check_monusac():
    train_loader = get_monusac_loader(
        path=MONUSAC_ROOT,
        download=True,
        patch_shape=(512, 512),
        batch_size=2,
        split="train",
        organ_type=["breast", "lung"]
    )
    print("Length of train loader: ", len(train_loader))
    check_loader(train_loader, 8, instance_labels=True, rgb=True, plt=True, save_path="./monusac_train.png")

    test_loader = get_monusac_loader(
        path=MONUSAC_ROOT,
        download=True,
        patch_shape=(512, 512),
        batch_size=1,
        split="test",
        organ_type=["breast", "prostate"]
    )
    print("Length of test loader: ", len(test_loader))
    check_loader(test_loader, 8, instance_labels=True, rgb=True, plt=True, save_path="./monusac_test.png")


if __name__ == "__main__":
    check_monusac()
