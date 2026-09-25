import os

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_puwf_av_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_puwf_av():
    loader = get_puwf_av_loader(
        path=os.path.join(DATA_ROOT, "puwf_av"),
        batch_size=2,
        patch_shape=(1024, 1024),
        resize_inputs=True,
        download=True,
    )
    check_loader(loader, 8, rgb=True, instance_labels=True, plt=True, save_path="test.png")


if __name__ == "__main__":
    check_puwf_av()
