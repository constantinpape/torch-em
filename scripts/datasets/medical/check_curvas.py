from torch_em.data import MinInstanceSampler
from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_curvas_loader


ROOT = "/media/anwai/ANWAI/data/curvas"


def check_curvas():
    loader = get_curvas_loader(
        path=ROOT,
        patch_shape=(1, 512, 512),
        batch_size=2,
        ndim=2,
        rater="1",
        resize_inputs=False,
        download=False,
        sampler=MinInstanceSampler()
    )

    check_loader(loader, 8)


if __name__ == "__main__":
    check_curvas()
