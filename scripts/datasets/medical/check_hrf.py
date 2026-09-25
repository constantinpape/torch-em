import os

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.hrf import get_hrf_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_hrf():
    loader = get_hrf_loader(
        path=os.path.join(DATA_ROOT, "hrf"),
        patch_shape=(512, 512),
        batch_size=2,
        resize_inputs=True,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_hrf()
