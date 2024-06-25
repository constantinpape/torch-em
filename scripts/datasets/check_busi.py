from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_busi_loader


ROOT = "/media/anwai/ANWAI/data/busi"


def check_busi():
    loader = get_busi_loader(
        path=ROOT,
        patch_shape=(512, 512),
        batch_size=2,
        category=None,
        resize_inputs=False,
        download=True,
    )

    check_loader(loader, 8)


if __name__ == "__main__":
    check_busi()
