from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_hil_toothseg_loader


ROOT = "/media/anwai/ANWAI/data/hil_toothseg"


def check_hil_toothseg():
    loader = get_hil_toothseg_loader(
        path=ROOT,
        patch_shape=(512, 512),
        batch_size=2,
        split="train",
        resize_inputs=False,
    )
    check_loader(loader, 8)


check_hil_toothseg()
