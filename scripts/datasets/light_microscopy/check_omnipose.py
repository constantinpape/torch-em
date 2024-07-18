from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.light_microscopy import get_omnipose_loader


ROOT = "/media/anwai/ANWAI/data/omnipose"


def check_omnipose():
    loader = get_omnipose_loader(
        path=ROOT,
        batch_size=1,
        patch_shape=(1024, 1024),
        split="train",
        data_choice="worm_high_res",
        sampler=MinInstanceSampler(),
    )
    check_loader(loader, 8)


check_omnipose()
