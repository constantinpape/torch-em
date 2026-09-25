import os

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.rexgroundingct import get_rexgroundingct_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_rexgroundingct():
    loader = get_rexgroundingct_loader(
        path=os.path.join(DATA_ROOT, "rexgroundingct"),
        batch_size=1,
        patch_shape=(32, 512, 512),
        split="val",
        max_cases=5,
        sampler=MinInstanceSampler(),
        download=True,
    )

    check_loader(loader, 4, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_rexgroundingct()
