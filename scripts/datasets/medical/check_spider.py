from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical import get_spider_loader


ROOT = "/media/anwai/ANWAI/data/spider"


def check_spider():
    loader = get_spider_loader(
        path=ROOT,
        patch_shape=(1, 512, 512),
        batch_size=2,
        sampler=MinInstanceSampler()
    )

    check_loader(loader, 8)


check_spider()
