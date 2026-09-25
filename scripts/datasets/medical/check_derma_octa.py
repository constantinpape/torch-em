import os

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_derma_octa_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_derma_octa():
    loader = get_derma_octa_loader(
        path=os.path.join(DATA_ROOT, "derma_octa"),
        batch_size=2,
        patch_shape=(512, 512),
        dim="2d",
        plexus="all",
        resize_inputs=True,
        download=True,
    )
    check_loader(loader, 8, rgb=True, instance_labels=True, plt=True, save_path="test.png")


if __name__ == "__main__":
    check_derma_octa()
