import os

from torch_em.data import MinForegroundSampler
from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_colonvessels_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_colonvessels():
    loader = get_colonvessels_loader(
        path=os.path.join(DATA_ROOT, "colonvessels"),
        patch_shape=(32, 512, 512),
        batch_size=2,
        vessel_type="arteries",
        download=True,
        sampler=MinForegroundSampler(min_fraction=0.0001),
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_colonvessels()
