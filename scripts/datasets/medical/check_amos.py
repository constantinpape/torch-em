from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical import get_amos_loader

ROOT = "/media/anwai/ANWAI/data/amos"


def check_amos():
    loader = get_amos_loader(
        path=ROOT,
        split="train",
        patch_shape=(1, 512, 512),
        modality="mri",
        ndim=2,
        batch_size=2,
        download=True,
        sampler=MinInstanceSampler(min_num_instances=3),
        resize_inputs=False,
    )
    check_loader(loader, 8)


if __name__ == "__main__":
    check_amos()
