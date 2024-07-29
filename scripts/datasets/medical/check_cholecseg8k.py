from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_cholecseg8k_loader


ROOT = "/media/anwai/ANWAI/data/cholecseg8k"


def get_cholecseg8k():
    loader = get_cholecseg8k_loader(
        path=ROOT,
        patch_shape=(512, 512),
        batch_size=2,
        split="train",
        resize_inputs=True,
        download=False,
    )
    check_loader(loader, 8)


if __name__ == "__main__":
    get_cholecseg8k()
