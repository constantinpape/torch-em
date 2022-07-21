from torch_em.data.datasets import get_uro_cell_loader
from torch_em.util.debug import check_loader


def check_uro_cell_loader(target):
    loader = get_uro_cell_loader("./data", target=target, download=True,
                                 batch_size=1, patch_shape=(32, 128, 128))
    check_loader(loader, n_samples=5, instance_labels=True)


if __name__ == "__main__":
    check_uro_cell_loader(target="mito")
