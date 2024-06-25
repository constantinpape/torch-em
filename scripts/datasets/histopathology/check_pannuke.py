from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_pannuke_loader


PANNUKE_ROOT = "/scratch/usr/nimanwai/data/pannuke/"


def check_pannuke():
    loader = get_pannuke_loader(
        path=PANNUKE_ROOT,
        batch_size=2,
        patch_shape=(1, 512, 512),
        ndim=2,
        download=True
    )
    check_loader(loader, 8, instance_labels=True, plt=False, rgb=True)


if __name__ == "__main__":
    check_pannuke()
