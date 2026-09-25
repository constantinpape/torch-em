from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_robust_mips_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_robust_mips():
    loader = get_robust_mips_loader(
        path=f"{DATA_ROOT}/robust_mips",
        batch_size=2,
        patch_shape=(512, 512),
        split="train",
        resize_inputs=True,
        download=True,
    )
    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_robust_mips()
