import os

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_openswisshcc_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_openswisshcc():
    loader = get_openswisshcc_loader(
        path=os.path.join(DATA_ROOT, "openswisshcc"),
        patch_shape=(1, 512, 512),
        batch_size=2,
        label_choice="lesion",
        resize_inputs=True,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./openswisshcc.png")


if __name__ == "__main__":
    check_openswisshcc()
