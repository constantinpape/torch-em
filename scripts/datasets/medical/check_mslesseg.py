import os

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_mslesseg_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_mslesseg():
    loader = get_mslesseg_loader(
        path=os.path.join(DATA_ROOT, "mslesseg"),
        patch_shape=(1, 224, 224),
        batch_size=2,
        modality="FLAIR",
        ndim=2,
        resize_inputs=True,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_mslesseg()
