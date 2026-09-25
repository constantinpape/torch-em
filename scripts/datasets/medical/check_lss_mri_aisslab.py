import os

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.lss_mri_aisslab import get_lss_mri_aisslab_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_lss_mri_aisslab():
    loader = get_lss_mri_aisslab_loader(
        path=os.path.join(DATA_ROOT, "lss_mri_aisslab"),
        batch_size=2,
        patch_shape=(512, 512),
        resize_inputs=True,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test_lss_mri_aisslab.png")


if __name__ == "__main__":
    check_lss_mri_aisslab()
