from torch_em.data.datasets import get_mouse_embryo_loader
from torch_em.util.debug import check_loader


def check_mouse_embryo():
    loader = get_mouse_embryo_loader("./data/mouse_embryo", "nuclei", "train", (8, 512, 512), 1, download=True)
    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_mouse_embryo()
