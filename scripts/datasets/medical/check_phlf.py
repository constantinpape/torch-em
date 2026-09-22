import os

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_phlf_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_phlf():
    loader = get_phlf_loader(
        path=os.path.join(DATA_ROOT, "phlf"),
        patch_shape=(1, 512, 512),
        batch_size=2,
        label_choice="liver",
        ndim=2,
        resize_inputs=True,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_phlf()
