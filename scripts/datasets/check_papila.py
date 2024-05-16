from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_papila_loader


ROOT = "/media/anwai/ANWAI/data/papila"


def check_papila():
    loader = get_papila_loader(
        path=ROOT,
        patch_shape=(256, 256),
        batch_size=2,
        resize_inputs=False,
        download=True,
    )

    check_loader(loader, 8)


if __name__ == "__main__":
    check_papila()
