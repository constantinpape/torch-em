from torch_em.data.datasets import get_nuc_mm_loader
from torch_em.util.debug import check_loader

NUC_MM_ROOT = "/scratch/usr/nimanwai/data/nuc_mm"


def check_nuc_mm():
    loader = get_nuc_mm_loader(
        NUC_MM_ROOT, "mouse", "train", patch_shape=(1, 192, 192), batch_size=1, download=True
    )
    check_loader(loader, 5, instance_labels=True, plt=True, save_path="mouse_loader.png")

    loader = get_nuc_mm_loader(
        NUC_MM_ROOT, "zebrafish", "train", patch_shape=(1, 64, 64), batch_size=1, download=True
    )
    check_loader(loader, 5, instance_labels=True, plt=True, save_path="zebrafish_loader.png")


if __name__ == "__main__":
    check_nuc_mm()
