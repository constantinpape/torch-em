from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_montgomery_loader


ROOT = "/media/anwai/ANWAI/data/montgomery"


def check_montgomery():
    loader = get_montgomery_loader(
        path=ROOT,
        patch_shape=(512, 512),
        batch_size=2,
        download=True,
    )

    check_loader(loader, 8)


if __name__ == "__main__":
    check_montgomery()
