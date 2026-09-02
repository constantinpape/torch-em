from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical import get_covid19_seg_loader


ROOT = "/media/anwai/ANWAI/data/covid19_seg"


def check_covid19_seg():
    loader = get_covid19_seg_loader(
        path=ROOT,
        patch_shape=(32, 512, 512),
        batch_size=2,
        task="lung",
        download=True,
        sampler=MinInstanceSampler(),
    )

    check_loader(loader, 8)


if __name__ == "__main__":
    check_covid19_seg()
