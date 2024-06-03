from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_toothfairy2_loader


ROOT = "/media/anwai/ANWAI/data/toothfairy2"


def check_toothfairy2():
    loader = get_toothfairy2_loader(
        path=ROOT,
        patch_shape=(1, 512, 512),
        batch_size=2,
        ndim=2,
        resize_inputs=False,
        download=False,
    )

    check_loader(loader, 8)


if __name__ == "__main__":
    check_toothfairy2()
