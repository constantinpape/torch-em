import os

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_mct_ltdiag_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_mct_ltdiag():
    loader = get_mct_ltdiag_loader(
        path=os.path.join(DATA_ROOT, "mct_ltdiag"),
        batch_size=1,
        patch_shape=(32, 512, 512),
        target="tumor",
        n_patients=4,
        download=True,
    )
    check_loader(loader, 4, plt=True, save_path="test.png")


if __name__ == "__main__":
    check_mct_ltdiag()
