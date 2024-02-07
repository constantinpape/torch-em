from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_autopet_loader

AUTOPET_ROOT = "/scratch/usr/nimanwai/test/data/"


def check_autopet():
    loader = get_autopet_loader(
        path=AUTOPET_ROOT,
        patch_shape=(1, 512, 512),
        batch_size=2,
        ndim=2,
        download=True,
    )
    print(f"Length of the loader: {len(loader)}")
    check_loader(loader, 8, plt=True, save_path="autopet.png")


if __name__ == "__main__":
    check_autopet()
