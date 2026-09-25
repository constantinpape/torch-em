import os

from torch_em.data import MinInstanceSampler
from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_recist_ct_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_recist_ct():
    loader = get_recist_ct_loader(
        path=os.path.join(DATA_ROOT, "recist_ct"),
        patch_shape=(32, 512, 512),
        batch_size=2,
        download=True,
        sampler=MinInstanceSampler(),
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_recist_ct()
