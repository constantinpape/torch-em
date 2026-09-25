from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_coph100_loader


ROOT = "/mnt/vast-kisski/home/archit/u28048/torch-em/scripts/datasets/data/coph100"


def check_coph100():
    loader = get_coph100_loader(
        path=ROOT,
        patch_shape=(512, 512),
        batch_size=2,
        resize_inputs=False,
        download=True,
    )

    check_loader(loader, 8, rgb=True)


if __name__ == "__main__":
    check_coph100()
