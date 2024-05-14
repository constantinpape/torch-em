from torch_em.util.debug import check_loader
from torch_em.data import MinForegroundSampler
from torch_em.data.datasets.medical import get_siim_acr_loader


ROOT = "/media/anwai/ANWAI/data/siim_acr"


def check_siim_acr():
    loader = get_siim_acr_loader(
        path=ROOT,
        split="train",
        patch_shape=(1024, 1024),
        batch_size=2,
        download=True,
        sampler=MinForegroundSampler(min_fraction=0.001)
    )
    check_loader(loader, 8)


if __name__ == "__main__":
    check_siim_acr()
