from torch_em.data.datasets import (
    get_neurips_cellseg_supervised_loader, get_neurips_cellseg_unsupervised_loader
)
from torch_em.util.debug import check_loader

NEURIPS_ROOT = "/home/pape/Work/data/neurips-cell-seg"


def check_neurips():
    loader = get_neurips_cellseg_supervised_loader(NEURIPS_ROOT, "train", (512, 512), 1)
    check_loader(loader, 15, instance_labels=True, rgb=True)

    loader = get_neurips_cellseg_unsupervised_loader(NEURIPS_ROOT, (512, 512), 1)
    check_loader(loader, 15, rgb=True)


if __name__ == "__main__":
    check_neurips()
