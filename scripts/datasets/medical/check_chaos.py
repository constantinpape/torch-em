from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical import get_chaos_loader


ROOT = "/media/anwai/ANWAI/data/chaos"


def check_chaos():
    loader = get_chaos_loader(
        path=ROOT,
        patch_shape=(32, 256, 256),
        batch_size=1,
        split="train",
        modality="MRI",
        download=True,
        sampler=MinInstanceSampler(min_num_instances=4)
    )

    check_loader(loader, 8)


if __name__ == "__main__":
    check_chaos()
