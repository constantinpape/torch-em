import os

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_cesarean_scar_defect_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_cesarean_scar_defect():
    loader = get_cesarean_scar_defect_loader(
        path=os.path.join(DATA_ROOT, "cesarean_scar_defect"),
        batch_size=2,
        patch_shape=(512, 512),
        split="train",
        resize_inputs=True,
        download=True,
    )
    check_loader(loader, 4, instance_labels=True, plt=True, save_path="test.png")


if __name__ == "__main__":
    check_cesarean_scar_defect()
