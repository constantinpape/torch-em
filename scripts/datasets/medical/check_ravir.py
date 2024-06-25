from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_ravir_loader


ROOT = "/media/anwai/ANWAI/data/ravir"


def check_ravir():
    loader = get_ravir_loader(
        path=ROOT,
        patch_shape=(256, 256),
        batch_size=2,
        resize_inputs=True,
        download=True,
    )

    check_loader(loader, 8)


if __name__ == "__main__":
    check_ravir()
