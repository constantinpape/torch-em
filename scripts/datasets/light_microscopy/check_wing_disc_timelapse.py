import os

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_wing_disc_timelapse_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_wing_disc_timelapse():
    loader = get_wing_disc_timelapse_loader(
        path=os.path.join(DATA_ROOT, "wing_disc_timelapse"),
        batch_size=2,
        patch_shape=(512, 512),
        movies=["Ecd_20141010_P2"],
        download=True,
    )
    check_loader(loader, 4, instance_labels=True, plt=True, save_path="test.png")


if __name__ == "__main__":
    check_wing_disc_timelapse()
