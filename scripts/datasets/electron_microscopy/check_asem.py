from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_asem_loader
from torch_em.data import MinForegroundSampler


ASEM_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data/asem"


def check_asem():
    # The volumes with small mito label crops (64, 2E) and the annotated block of 61 download in minutes.
    loader = get_asem_loader(
        path=ASEM_ROOT,
        patch_shape=(1, 192, 192),
        batch_size=2,
        ndim=2,
        download=True,
        organelles="mito",
        volume_ids=["61", "64", "2E"],
        sampler=MinForegroundSampler(min_fraction=0.01),
    )
    print(f"Length of the loader: {len(loader)}")
    check_loader(loader, 8, instance_labels=False, plt=True, rgb=False, save_path="./asem_loader.png")


if __name__ == "__main__":
    check_asem()
