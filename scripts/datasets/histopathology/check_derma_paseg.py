from torch_em.util.debug import check_loader
from torch_em.data.datasets.histopathology.derma_paseg import get_derma_paseg_loader

DATA_PATH = "/tmp/derma_paseg_test"


def check_derma_paseg():
    loader = get_derma_paseg_loader(
        DATA_PATH,
        batch_size=1,
        patch_shape=(512, 512),
        split="train",
        stain="unstained",
        download=True,
        shuffle=True,
    )
    check_loader(loader, 8, rgb=True)


if __name__ == "__main__":
    check_derma_paseg()
