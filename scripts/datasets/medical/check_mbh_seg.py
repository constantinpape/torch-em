from torch_em.data import MinInstanceSampler
from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_mbh_seg_loader

ROOT = "/media/anwai/ANWAI/data/mbh-seg"


def check_mbh_seg():
    loader = get_mbh_seg_loader(
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
    check_mbh_seg()
