import os

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.longitudinal_ct import get_longitudinal_ct_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_longitudinal_ct():
    loader = get_longitudinal_ct_loader(
        path=os.path.join(DATA_ROOT, "longitudinal_ct"),
        batch_size=1,
        patch_shape=(32, 256, 256),
        ndim=3,
        sampler=MinInstanceSampler(),
        download=False,
    )

    check_loader(loader, 8, plt=True, save_path="./test_longitudinal_ct.png")


if __name__ == "__main__":
    check_longitudinal_ct()
