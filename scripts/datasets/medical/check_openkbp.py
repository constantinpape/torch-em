import os

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.openkbp import get_openkbp_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_openkbp():
    loader = get_openkbp_loader(
        path=os.path.join(DATA_ROOT, "openkbp"),
        patch_shape=(64, 64, 64),
        batch_size=1,
        split="train",
        sampler=MinInstanceSampler(),
        ndim=3,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_openkbp()
