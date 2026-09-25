import os

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_far_polyp_seg_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_far_polyp_seg():
    loader = get_far_polyp_seg_loader(
        path=os.path.join(DATA_ROOT, "far_polyp_seg"),
        patch_shape=(512, 512),
        batch_size=2,
        resize_inputs=True,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./far_polyp_seg.png")


if __name__ == "__main__":
    check_far_polyp_seg()
