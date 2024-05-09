from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_jsrt_loader


ROOT = "/media/anwai/ANWAI/data/jsrt"


def check_jsrt():
    loader = get_jsrt_loader(
        path=ROOT,
        split="test",
        patch_shape=(256, 256),
        batch_size=2,
        choice=None,
        download=True,
    )

    check_loader(loader, 8)


if __name__ == "__main__":
    check_jsrt()
