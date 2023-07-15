from torch_em.data.datasets import get_lizard_loader
from torch_em.util.debug import check_loader


def check_lizard():
    loader = get_lizard_loader("./data/lizard", (512, 512), 1, download=True)
    check_loader(loader, 8, rgb=True, instance_labels=True)


if __name__ == "__main__":
    check_lizard()
