from torch_em.util.debug import check_loader
from torch_em.data.datasets.light_microscopy import get_cellpose_loader


ROOT = "/media/anwai/ANWAI/data/cellpose/"


def check_cellpose():
    loader = get_cellpose_loader(
        path=ROOT,
        split="train",
        patch_shape=(512, 512),
        batch_size=1,
        choice="cyto",
    )
    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_cellpose()
