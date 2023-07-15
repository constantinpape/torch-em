from torch_em.data.datasets import get_uro_cell_loader
from torch_em.util.debug import check_loader


def check_uro_cell():
    loader = get_uro_cell_loader("./data/uro_cell", "mito", (8, 512, 512), 1, download=True)
    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_uro_cell()
