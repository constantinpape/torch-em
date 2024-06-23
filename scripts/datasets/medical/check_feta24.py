from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_feta24_loader


ROOT = "/media/anwai/ANWAI/data/feta24"


def check_feta24():
    loader = get_feta24_loader(
        path=ROOT,
        patch_shape=(1, 512, 512),
        batch_size=2,
        resize_inputs=True,
        download=False,
    )

    check_loader(loader, 8)


if __name__ == "__main__":
    check_feta24()
