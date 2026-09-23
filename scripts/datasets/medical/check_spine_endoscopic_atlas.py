import os

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_spine_endoscopic_atlas_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_spine_endoscopic_atlas():
    loader = get_spine_endoscopic_atlas_loader(
        path=os.path.join(DATA_ROOT, "spine_endoscopic_atlas"),
        batch_size=1,
        patch_shape=(512, 512),
        resize_inputs=True,
        download=True,
    )
    check_loader(loader, 4, instance_labels=True, plt=True, save_path="test.png")


if __name__ == "__main__":
    check_spine_endoscopic_atlas()
