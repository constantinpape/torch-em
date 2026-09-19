import os

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_les_av_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_les_av():
    loader = get_les_av_loader(
        path=os.path.join(DATA_ROOT, "les_av"),
        batch_size=2,
        patch_shape=(512, 512),
        resize_inputs=True,
        download=True,
    )
    check_loader(loader, 8, instance_labels=True, plt=True, save_path="./les_av.png")


if __name__ == "__main__":
    check_les_av()
