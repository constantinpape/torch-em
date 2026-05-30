import matplotlib
matplotlib.use("Agg")

from torch_em.data.datasets import get_mousecc_loader  # noqa
from torch_em.util.debug import check_loader  # noqa

DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data/mousecc"


def check_mousecc(label_choice="myelin"):
    loader = get_mousecc_loader(
        DATA_ROOT, patch_shape=(1, 512, 512), batch_size=1, label_choice=label_choice, ndim=2
    )
    check_loader(loader, 8, plt=True, save_path=f"./check_mousecc_{label_choice}.png")


def main():
    check_mousecc("myelin")
    check_mousecc("fibers")


if __name__ == "__main__":
    main()
