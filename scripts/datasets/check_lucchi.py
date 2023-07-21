from torch_em.data.datasets import get_lucchi_loader
from torch_em.util.debug import check_loader


def check_lucchi():
    loader = get_lucchi_loader("./data/lucchi", "train", (8, 512, 512), 1, download=True)
    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_lucchi()
