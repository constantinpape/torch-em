import os

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.idrid import get_idrid_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_refined_idrid():
    loader = get_idrid_loader(
        path=os.path.join(DATA_ROOT, "refined_idrid"),
        batch_size=1,
        patch_shape=(512, 512),
        split="train",
        resize_inputs=True,
        sampler=MinInstanceSampler(),
        download=True,
        version="refined",
    )

    check_loader(loader, 8, plt=True, save_path="./test_refined_idrid.png")


if __name__ == "__main__":
    check_refined_idrid()
