import os

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.tn3k import get_tn3k_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_tn3k():
    loader = get_tn3k_loader(
        path=os.path.join(DATA_ROOT, "tn3k"),
        patch_shape=(512, 512),
        batch_size=2,
        split="trainval",
        resize_inputs=False,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_tn3k()
