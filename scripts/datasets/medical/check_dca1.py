from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_dca1_loader


ROOT = "/media/anwai/ANWAI/data/dca1"


def check_dca1():
    loader = get_dca1_loader(
        path=ROOT,
        patch_shape=(512, 512),
        batch_size=2,
        split="test",
        resize_inputs=True,
        download=False,
    )
    check_loader(loader, 8)


if __name__ == "__main__":
    check_dca1()
