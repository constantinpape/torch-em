import os

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.trusted import get_trusted_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_trusted():
    loader = get_trusted_loader(
        path=os.path.join(DATA_ROOT, "trusted"),
        patch_shape=(32, 256, 256),
        batch_size=1,
        modality="us",
        label_choice="gt_estimated",
        sampler=MinInstanceSampler(),
        download=True,
    )

    check_loader(loader, 4, plt=True, save_path="./test_trusted.png")


if __name__ == "__main__":
    check_trusted()
