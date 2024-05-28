from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_nih_lymph_ct_loader


ROOT = "/media/anwai/ANWAI/data/nih_lymph_ct/"


def check_nih_lymph_ct():
    loader = get_nih_lymph_ct_loader(
        path=ROOT,
        patch_shape=(1, 512, 512),
        batch_size=2,
        download=True,
    )

    check_loader(loader, 8)


if __name__ == "__main__":
    check_nih_lymph_ct()
