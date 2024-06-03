from torch_em.data import MinInstanceSampler
from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_acouslic_ai_loader


ROOT = "/media/anwai/ANWAI/data/acouslic_ai"


def check_acouslic_ai():
    loader = get_acouslic_ai_loader(
        path=ROOT,
        patch_shape=(1, 512, 512),
        ndim=2,
        batch_size=2,
        resize_inputs=False,
        download=False,
        sampler=MinInstanceSampler()
    )

    check_loader(loader, 8)


if __name__ == "__main__":
    check_acouslic_ai()
