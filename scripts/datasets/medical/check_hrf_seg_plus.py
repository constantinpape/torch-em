import os

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.hrf_seg_plus import get_hrf_seg_plus_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_hrf_seg_plus():
    loader = get_hrf_seg_plus_loader(
        path=os.path.join(DATA_ROOT, "hrf_seg_plus"),
        batch_size=2,
        patch_shape=(500, 500),
        resize_inputs=False,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test_hrf_seg_plus.png")


if __name__ == "__main__":
    check_hrf_seg_plus()
