from torch_em.data.datasets import get_axondeepseg_loader
from torch_em.util.debug import check_loader


def check_axondeepseg():
    loader = get_axondeepseg_loader(
        "./data/axondeepseg", name="sem", patch_shape=(1024, 1024), batch_size=1,
        one_hot_encoding=True, shuffle=True, download=True
    )
    check_loader(loader, 5)

    loader = get_axondeepseg_loader(
        "./data/axondeepseg", name="tem", patch_shape=(1024, 1024), batch_size=1,
        one_hot_encoding=True, shuffle=True, download=True
    )
    check_loader(loader, 5)


if __name__ == "__main__":
    check_axondeepseg()
