from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_asem_loader
from torch_em.data import MinForegroundSampler


# ASEM_ROOT = "/media/anwai/ANWAI/data/asem"
ASEM_ROOT = "/scratch/projects/nim00007/sam/data/asem"


def check_asem():
    loader = get_asem_loader(
        path=ASEM_ROOT,
        patch_shape=(1, 512, 512),
        batch_size=2,
        ndim=2,
        download=True,
        volume_ids="cell_1",
        organelles="er",
        sampler=MinForegroundSampler(min_fraction=0.01)
    )
    print(f"Length of the loader: {len(loader)}")
    check_loader(loader, 8, instance_labels=False, plt=True, rgb=False, save_path="./asem_loader.png")


if __name__ == "__main__":
    check_asem()
