import os

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_c3ro_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_c3ro():
    loader = get_c3ro_loader(
        path=os.path.join(DATA_ROOT, "c3ro"),
        batch_size=2,
        patch_shape=(32, 128, 128),
        site="H&N",
        annotator="expert",
        download=True,
    )
    check_loader(loader, 4, plt=True, save_path="test.png")


if __name__ == "__main__":
    check_c3ro()
