import os

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.deep_psma import get_deep_psma_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_deep_psma():
    loader = get_deep_psma_loader(
        path=os.path.join(DATA_ROOT, "deep_psma_test"),
        batch_size=1,
        patch_shape=(1, 512, 512),
        tracer="psma",
        modality="PET",
        ndim=2,
        sampler=MinInstanceSampler(),
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_deep_psma()
