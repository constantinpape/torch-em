from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_acdc_loader
from torch_em.data import MinInstanceSampler


ROOT = "/media/anwai/ANWAI/data/acdc"


def check_acdc():
    loader = get_acdc_loader(
        path=ROOT,
        patch_shape=(4, 256, 256),
        batch_size=2,
        split="train",
        download=True,
        sampler=MinInstanceSampler(min_num_instances=4),
    )

    check_loader(loader, 8)


if __name__ == "__main__":
    check_acdc()
