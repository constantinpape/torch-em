from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_m2caiseg_loader


ROOT = "/media/anwai/ANWAI/data/m2caiseg"


def check_m2caiseg():
    loader = get_m2caiseg_loader(
        path=ROOT,
        split="train",
        patch_shape=(512, 512),
        batch_size=2,
        resize_inputs=False,
        download=True,
    )

    check_loader(loader, 8)


if __name__ == "__main__":
    check_m2caiseg()
