from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_oimhs_loader


ROOT = "/media/anwai/ANWAI/data/oimhs"


def check_oimhs():
    loader = get_oimhs_loader(
        path=ROOT,
        patch_shape=(512, 512),
        batch_size=2,
        download=False,
        resize_inputs=True,
    )

    check_loader(loader, 8)


if __name__ == "__main__":
    check_oimhs()
