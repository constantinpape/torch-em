import os

from torch_em.data import MinForegroundSampler
from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_airrc_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_airrc():
    loader = get_airrc_loader(
        path=os.path.join(DATA_ROOT, "airrc"),
        patch_shape=(1, 224, 224),
        batch_size=2,
        subset_ids=[0],
        ndim=2,
        resize_inputs=True,
        download=True,
        sampler=MinForegroundSampler(min_fraction=0.001),
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_airrc()
