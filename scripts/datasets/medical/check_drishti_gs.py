import os

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.drishti_gs import get_drishti_gs_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_drishti_gs():
    loader = get_drishti_gs_loader(
        path=os.path.join(DATA_ROOT, "drishti_gs"),
        patch_shape=(512, 512),
        batch_size=2,
        split="train",
        task="optic_disc",
        resize_inputs=True,
        sampler=MinInstanceSampler(),
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test_drishti_gs.png")


if __name__ == "__main__":
    check_drishti_gs()
