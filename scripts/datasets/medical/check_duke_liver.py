from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_duke_liver_loader


ROOT = "/media/anwai/ANWAI/data/duke_liver"


def check_duke_liver():
    from micro_sam.training import identity
    loader = get_duke_liver_loader(
        path=ROOT,
        patch_shape=(32, 512, 512),
        batch_size=2,
        split="train",
        download=False,
        raw_transform=identity,

    )
    check_loader(loader, 8)


if __name__ == "__main__":
    check_duke_liver()
