from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_chaos_loader


ROOT = "/media/anwai/ANWAI/data/chaos"


def check_chaos():
    loader = get_chaos_loader(
        path=ROOT,
        patch_shape=(1, 512, 512),
        batch_size=2,
        split="train",
        modality="CT",
        resize_inputs=False,
        download=True,
    )

    check_loader(loader, 8)


if __name__ == "__main__":
    check_chaos()
