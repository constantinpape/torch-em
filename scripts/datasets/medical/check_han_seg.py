from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_han_seg_loader


ROOT = "/media/anwai/ANWAI/data/han-seg/"


def check_han_seg():
    loader = get_han_seg_loader(
        path=ROOT,
        patch_shape=(32, 512, 512),
        batch_size=2,
        download=False,
    )

    check_loader(loader, 8)


if __name__ == "__main__":
    check_han_seg()
