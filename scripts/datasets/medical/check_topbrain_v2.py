import os

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.topbrain import get_topbrain_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_topbrain_v2():
    loader = get_topbrain_loader(
        path=os.path.join(DATA_ROOT, "topbrain_v2"),
        patch_shape=(1, 512, 512),
        batch_size=1,
        modality="ct",
        version="v2",
        sampler=MinInstanceSampler(),
        ndim=2,
        download=False,
    )

    check_loader(loader, 8, plt=True, save_path="./test_topbrain_v2.png")


if __name__ == "__main__":
    check_topbrain_v2()
