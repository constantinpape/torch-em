import os

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.pants import get_pants_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_pants():
    loader = get_pants_loader(
        path=os.path.join(DATA_ROOT, "pants"),
        batch_size=1,
        patch_shape=(32, 256, 256),
        split="train",
        case_ids=["PanTS_00000684", "PanTS_00000710", "PanTS_00000767"],
        ndim=3,
        sampler=MinInstanceSampler(),
        download=False,
    )

    check_loader(loader, 3, plt=True, save_path="./test_pants.png")


if __name__ == "__main__":
    check_pants()
