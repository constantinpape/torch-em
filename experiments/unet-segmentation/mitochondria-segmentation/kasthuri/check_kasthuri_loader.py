from torch_em.data.datasets import get_kasthuri_loader
from torch_em.util.debug import check_loader


def check_kasthuri_loader(split):
    loader = get_kasthuri_loader("./data", split=split, download=True, batch_size=1, patch_shape=(64, 256, 256))
    check_loader(loader, n_samples=4, instance_labels=True)


if __name__ == "__main__":
    check_kasthuri_loader(split="train")
    check_kasthuri_loader(split="test")
