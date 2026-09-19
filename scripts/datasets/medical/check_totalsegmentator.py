import os

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.totalsegmentator import get_totalsegmentator_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_totalsegmentator(version):
    loader = get_totalsegmentator_loader(
        path=os.path.join(DATA_ROOT, f"totalsegmentator_{version}"),
        patch_shape=(1, 256, 256),
        batch_size=1,
        split="train",
        ndim=2,
        download=True,
        version=version,
        sampler=MinInstanceSampler(min_num_instances=3),
    )

    check_loader(loader, 8, plt=True, save_path=f"./test_{version}.png")


if __name__ == "__main__":
    check_totalsegmentator(version="v2")
    check_totalsegmentator(version="v3")
