import sys

from torch_em.data.datasets import get_liconn_loader
from torch_em.util.debug import check_loader

sys.path.append("..")

DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data/liconn"


def check_liconn():
    for seg in ("proofread", "agglomerated"):
        print(f"Checking segmentation: {seg}")
        loader = get_liconn_loader(
            DATA_ROOT, batch_size=1, patch_shape=(32, 256, 256), segmentation=seg, download=True
        )
        check_loader(loader, 4, instance_labels=True, plt=True, save_path=f"check_liconn_{seg}.png")


if __name__ == "__main__":
    check_liconn()
