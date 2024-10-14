from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_jnuifm_loader


ROOT = "/media/anwai/ANWAI/data/jnu-ifm"


def check_jnuifm():
    loader = get_jnuifm_loader(
        path=ROOT,
        patch_shape=(512, 512),
        batch_size=2,
        resize_inputs=True,
        download=True,
    )

    check_loader(loader, 8)


if __name__ == "__main__":
    check_jnuifm()
