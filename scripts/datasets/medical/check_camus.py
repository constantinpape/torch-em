from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_camus_loader


ROOT = "/media/anwai/ANWAI/data/camus"


def check_camus():
    loader = get_camus_loader(
        path=ROOT,
        patch_shape=(1, 512, 512),
        batch_size=2,
        chamber=2,
        resize_inputs=True,
        download=True,
    )
    check_loader(loader, 8)


if __name__ == "__main__":
    check_camus()
