import os

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.tg3k import get_tg3k_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_tg3k():
    loader = get_tg3k_loader(
        path=os.path.join(DATA_ROOT, "tn3k"),  # TG3K ships in the same archive as TN3K, already downloaded there.
        patch_shape=(512, 512),
        batch_size=2,
        resize_inputs=False,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_tg3k()
