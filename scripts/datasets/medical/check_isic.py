from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_isic_loader


ROOT = "/scratch/share/cidas/cca/data/isic"


def check_isic():
    loader = get_isic_loader(
        path=ROOT,
        patch_shape=(700, 700),
        batch_size=2,
        split="test",
        download=True,
        resize_inputs=True,
    )

    check_loader(loader, 8, plt=True, save_path="./isic.png")


if __name__ == "__main__":
    check_isic()
