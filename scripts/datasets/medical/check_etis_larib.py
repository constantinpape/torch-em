import os

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.etis_larib import get_etis_larib_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_etis_larib():
    loader = get_etis_larib_loader(
        path=os.path.join(DATA_ROOT, "etis_larib"),
        patch_shape=(512, 512),
        batch_size=2,
        resize_inputs=True,
        sampler=MinInstanceSampler(),
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test_etis_larib.png")


if __name__ == "__main__":
    check_etis_larib()
