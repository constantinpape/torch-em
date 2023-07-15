from torch_em.data.datasets import get_sponge_em_loader
from torch_em.util.debug import check_loader


def check_sponge_em():
    loader = get_sponge_em_loader("./data/sponge_em", "instances", (8, 512, 512), 1, download=True)
    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_sponge_em()
