import os

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.hqcolon import get_hqcolon_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_hqcolon():
    loader = get_hqcolon_loader(
        path=os.path.join(DATA_ROOT, "hqcolon"),
        patch_shape=(32, 512, 512),
        batch_size=1,
        label_choice="gas_and_fluid",
        sampler=MinInstanceSampler(),
        download=True,
    )

    check_loader(loader, 4, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_hqcolon()
