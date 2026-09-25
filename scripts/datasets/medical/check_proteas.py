import os

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_proteas_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_proteas():
    loader = get_proteas_loader(
        path=os.path.join(DATA_ROOT, "proteas"),
        batch_size=1,
        patch_shape=(32, 128, 128),
        sequence="t1c",
        n_patients=4,
        download=True,
    )
    check_loader(loader, 4, instance_labels=True, plt=True, save_path="test.png")


if __name__ == "__main__":
    check_proteas()
