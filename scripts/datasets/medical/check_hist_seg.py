import os

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_hist_seg_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_hist_seg():
    loader = get_hist_seg_loader(
        path=os.path.join(DATA_ROOT, "hist_seg"),
        batch_size=2,
        patch_shape=(256, 256),
        n_cases=3,
        download=True,
    )
    check_loader(loader, 4, instance_labels=False, plt=True, save_path="test.png")


if __name__ == "__main__":
    check_hist_seg()
