from torch_em.data.datasets import get_pnas_membrane_loader, get_pnas_nucleus_loader
from torch_em.util.debug import check_loader


def check_membrane_loader():
    loader = get_pnas_membrane_loader("./data", plant_ids=None, download=True, batch_size=1, patch_shape=(64, 256, 256))
    check_loader(loader, n_samples=4, instance_labels=True)


def check_nucleus_loader():
    loader = get_pnas_nucleus_loader("./data", plant_ids=None, download=True, batch_size=1, patch_shape=(64, 256, 256))
    check_loader(loader, n_samples=4, instance_labels=True)


if __name__ == "__main__":
    check_membrane_loader()
    check_nucleus_loader()
