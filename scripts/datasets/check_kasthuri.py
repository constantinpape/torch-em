from torch_em.data.datasets import get_kasthuri_loader
from torch_em.util.debug import check_loader


def check_kasthuri():
    loader = get_kasthuri_loader("./data/kasthuri", "train", (8, 512, 512), 1, download=True)
    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_kasthuri()
