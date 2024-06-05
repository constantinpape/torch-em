from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_uwaterloo_skin_loader


ROOT = "/media/anwai/ANWAI/data/uwaterloo_skinseg"


def check_uwaterloo_skin():
    loader = get_uwaterloo_skin_loader(
        path=ROOT,
        patch_shape=(512, 512),
        batch_size=2,
        resize_inputs=True,
        download=True,
    )

    check_loader(loader, 8)


if __name__ == "__main__":
    check_uwaterloo_skin()
