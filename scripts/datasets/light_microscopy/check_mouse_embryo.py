from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_mouse_embryo_loader


ROOT = "/media/anwai/ANWAI/data/mouse_embryo"


def check_mouse_embryo():
    loader = get_mouse_embryo_loader(
        path=ROOT,
        name="nuclei",
        split="train",
        patch_shape=(8, 512, 512),
        batch_size=1,
        download=True,
        with_padding_per_patch=True,
    )
    check_loader(loader, 8, instance_labels=True)

    loader = get_mouse_embryo_loader(
        path=ROOT,
        name="membrane",
        split="train",
        patch_shape=(8, 512, 512),
        batch_size=1,
        download=True,
        with_padding_per_patch=True,
    )
    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_mouse_embryo()
