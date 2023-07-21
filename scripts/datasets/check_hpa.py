from torch_em.data.datasets import get_hpa_segmentation_loader
from torch_em.util.debug import check_loader


def check_hpa():
    loader = get_hpa_segmentation_loader("./data/hpa", "train", (512, 512), 1, download=True)
    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_hpa()
