import os

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.kvasir_instrument import get_kvasir_instrument_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_kvasir_instrument():
    loader = get_kvasir_instrument_loader(
        path=os.path.join(DATA_ROOT, "kvasir_instrument"),
        patch_shape=(512, 512),
        batch_size=2,
        split="test",
        resize_inputs=True,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_kvasir_instrument()
