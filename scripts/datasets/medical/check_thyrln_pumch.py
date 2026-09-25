from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_thyrln_pumch_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_thyrln_pumch():
    loader = get_thyrln_pumch_loader(
        path=f"{DATA_ROOT}/thyrln_pumch",
        batch_size=2,
        patch_shape=(512, 512),
        resize_inputs=True,
        download=True,
    )
    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_thyrln_pumch()
