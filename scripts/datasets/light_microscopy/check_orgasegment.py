from torch_em.util.debug import check_loader
from torch_em.data.datasets.light_microscopy import get_orgasegment_loader


ROOT = "/media/anwai/ANWAI/data/orgasegment"


def check_orgasegment():
    loader = get_orgasegment_loader(
        path=ROOT,
        split="val",
        patch_shape=(512, 512),
        batch_size=1,
        download=True,
    )
    check_loader(loader, 8, instance_labels=True)


def main():
    check_orgasegment()


if __name__ == "__main__":
    main()
