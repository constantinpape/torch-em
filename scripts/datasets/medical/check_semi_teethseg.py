import os

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.semi_teethseg import get_semi_teethseg_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_semi_teethseg():
    loader = get_semi_teethseg_loader(
        path=os.path.join(DATA_ROOT, "semi_teethseg"),
        batch_size=2,
        patch_shape=(512, 512),
        split="adult",
        resize_inputs=True,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_semi_teethseg()
