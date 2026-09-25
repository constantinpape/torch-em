import os

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_ima_plus_plus_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_ima_plus_plus():
    # Restrict to the small "A09" annotator subset (93 masks) to keep the per-image
    # ISIC download in this check manageable.
    loader = get_ima_plus_plus_loader(
        path=os.path.join(DATA_ROOT, "ima_plus_plus"),
        patch_shape=(512, 512),
        batch_size=2,
        annotator="A09",
        resize_inputs=True,
        download=True,
    )

    check_loader(loader, 8, rgb=True, plt=True, save_path="./ima_plus_plus.png")


if __name__ == "__main__":
    check_ima_plus_plus()
