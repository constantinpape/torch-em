import os

import matplotlib
matplotlib.use("Agg")

from torch_em.data.datasets import get_hydra_loader  # noqa
from torch_em.data.sampler import MinInstanceSampler  # noqa
from torch_em.util.debug import check_loader  # noqa

CIDAS_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_hydra():
    loader = get_hydra_loader(
        os.path.join(CIDAS_ROOT, "hydra"), batch_size=1, patch_shape=(32, 256, 256),
        download=True, image_mip=3, seg_mip=2, sampler=MinInstanceSampler(min_num_instances=2)
    )
    check_loader(loader, 8, instance_labels=True, plt=True, save_path="./check_hydra_vulgaris.png")


def main():
    check_hydra()


if __name__ == "__main__":
    main()
