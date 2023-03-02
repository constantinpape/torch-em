from torch_em.data.datasets import get_neurips_cellseg_supervised_loader, get_neurips_cellseg_unsupervised_loader
from torch_em.util.debug import check_loader


def check_supervised(split, n_images=6):
    root = "/home/pape/Work/data/neurips-cell-seg"
    patch_shape = [384, 384]
    loader = get_neurips_cellseg_supervised_loader(root, split, patch_shape, batch_size=1)
    check_loader(loader, n_images, instance_labels=True, rgb=True)


def check_unsupervised(n_images=10):
    root = "/home/pape/Work/data/neurips-cell-seg"
    patch_shape = [384, 384]
    loader = get_neurips_cellseg_unsupervised_loader(root, patch_shape, batch_size=1,
                                                     use_images=True, use_wholeslide=True)
    check_loader(loader, n_images, rgb=True)


def main():
    check_supervised("train")
    check_supervised("val")

    check_unsupervised()


if __name__ == "__main__":
    main()
