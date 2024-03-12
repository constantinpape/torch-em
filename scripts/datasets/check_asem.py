from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_asem_loader


ASEM_ROOT = "/media/anwai/ANWAI/data/asem"


def check_asem():
    loader = get_asem_loader(
        path=ASEM_ROOT,
        patch_shape=(1, 512, 512),
        batch_size=2,
        ndim=2,
        download=True
    )
    print(f"Length of the loader: {len(loader)}")
    check_loader(loader, 8)


if __name__ == "__main__":
    check_asem()
