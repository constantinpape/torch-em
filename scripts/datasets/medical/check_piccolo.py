from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_piccolo_loader


ROOT = "/media/anwai/ANWAI/data/piccolo"


def check_piccolo():
    loader = get_piccolo_loader(
        path=ROOT,
        patch_shape=(512, 512),
        batch_size=2,
        split="train",
        resize_inputs=True,
    )
    check_loader(loader, 8)


if __name__ == "__main__":
    check_piccolo()
