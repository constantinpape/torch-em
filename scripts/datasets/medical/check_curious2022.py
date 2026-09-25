import os

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.curious2022 import get_curious2022_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_curious2022():
    for task in ["tumor", "resection"]:
        loader = get_curious2022_loader(
            path=os.path.join(DATA_ROOT, "curious2022"),
            batch_size=1,
            patch_shape=(32, 256, 256),
            task=task,
            ndim=3,
            sampler=MinInstanceSampler(),
            download=True,
        )
        check_loader(loader, 4, plt=True, save_path=f"./test_curious2022_{task}.png")


if __name__ == "__main__":
    check_curious2022()
