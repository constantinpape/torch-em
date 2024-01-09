from torch_em.data.datasets import get_axondeepseg_loader
from torch_em.util.debug import check_loader


ROOT = "/scratch/usr/nimanwai/data/axondeepseg"


def check_axondeepseg():
    loader = get_axondeepseg_loader(
        ROOT, name="sem", patch_shape=(1024, 1024), batch_size=1, split="train",
        one_hot_encoding=True, shuffle=True, download=True, val_fraction=0.1
    )
    check_loader(loader, 5, True, True, False, "sem_loader.png")

    loader = get_axondeepseg_loader(
        ROOT, name="tem", patch_shape=(1024, 1024), batch_size=1, split="train",
        one_hot_encoding=True, shuffle=True, download=True, val_fraction=0.1
    )
    check_loader(loader, 5, True, True, False, "tem_loader.png")


if __name__ == "__main__":
    check_axondeepseg()
