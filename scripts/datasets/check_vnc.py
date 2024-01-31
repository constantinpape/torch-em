from torch_em.data.datasets import get_vnc_loader
from torch_em.util.debug import check_loader


def check_vnc():
    loader = get_vnc_loader("./data/vnc", "neurons", (8, 512, 512), 1, download=True)
    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_vnc()
