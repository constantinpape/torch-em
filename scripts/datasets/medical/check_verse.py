from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_verse_loader


ROOT = "/media/anwai/ANWAI/data/verse"


def check_verse():
    loader = get_verse_loader(
        path=ROOT,
        split="test",
        patch_shape=(1, 512, 512),
        batch_size=2,
        download=True,
        resize_inputs=False,
    )

    check_loader(loader, 8)


if __name__ == "__main__":
    check_verse()
