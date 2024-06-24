from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_osic_pulmofib_loader


ROOT = "/media/anwai/ANWAI/data/osic_pulmofib"


def check_osic_pulmofib():
    loader = get_osic_pulmofib_loader(
        path=ROOT,
        patch_shape=(1, 512, 512),
        batch_size=2,
        resize_inputs=False,
        download=True,
    )

    check_loader(loader, 8)


if __name__ == "__main__":
    check_osic_pulmofib()
