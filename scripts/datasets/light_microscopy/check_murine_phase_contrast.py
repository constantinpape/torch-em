import os

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_murine_phase_contrast_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_murine_phase_contrast():
    loader = get_murine_phase_contrast_loader(
        path=os.path.join(DATA_ROOT, "murine_phase_contrast"),
        batch_size=2,
        patch_shape=(512, 512),
        split="train",
        download=True,
    )
    check_loader(loader, 4, instance_labels=True, plt=True, save_path="test.png")


if __name__ == "__main__":
    check_murine_phase_contrast()
