import os

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_pmcanalseg_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_pmcanalseg():
    loader = get_pmcanalseg_loader(
        path=os.path.join(DATA_ROOT, "pmcanalseg"),
        patch_shape=(32, 256, 256),
        batch_size=2,
        label_choice="mandibular",
        resize_inputs=True,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./pmcanalseg.png")


if __name__ == "__main__":
    check_pmcanalseg()
