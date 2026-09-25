import os

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.cvc_colondb import get_cvc_colondb_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_cvc_colondb():
    loader = get_cvc_colondb_loader(
        path=os.path.join(DATA_ROOT, "cvc_colondb"),
        patch_shape=(512, 512),
        batch_size=2,
        resize_inputs=True,
        sampler=MinInstanceSampler(),
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_cvc_colondb()
