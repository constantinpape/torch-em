import os

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_csc_fundus_segmentation_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_csc_fundus_segmentation():
    loader = get_csc_fundus_segmentation_loader(
        path=os.path.join(DATA_ROOT, "csc_fundus_segmentation"),
        patch_shape=(512, 512),
        batch_size=2,
        split="test",
        grader="grader1",
        resize_inputs=True,
        download=True,
    )

    check_loader(loader, 8, rgb=True, plt=True, save_path="./csc_fundus_segmentation.png")


if __name__ == "__main__":
    check_csc_fundus_segmentation()
