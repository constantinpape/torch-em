import os

from torch_em.data import MinForegroundSampler
from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_rvo_me_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_rvo_me():
    loader = get_rvo_me_loader(
        path=os.path.join(DATA_ROOT, "rvo_me"),
        patch_shape=(384, 384),
        batch_size=2,
        split="train",
        resize_inputs=True,
        download=True,
        sampler=MinForegroundSampler(min_fraction=0.001),
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_rvo_me()
