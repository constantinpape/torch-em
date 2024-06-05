from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_idrid_loader


URL = "aaryapatel98/indian-diabetic-retinopathy-image-dataset"
ROOT = "/media/anwai/ANWAI/data/idrid"


def check_idrid():
    loader = get_idrid_loader(
        path=ROOT,
        patch_shape=(512, 512),
        batch_size=2,
        split="train",
        task="soft_exudates",
        resize_inputs=True,
        download=True,
    )

    check_loader(loader, 8)


if __name__ == "__main__":
    check_idrid()
