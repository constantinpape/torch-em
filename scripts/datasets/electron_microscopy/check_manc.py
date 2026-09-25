import os

import matplotlib
matplotlib.use("Agg")

from torch_em.data.datasets.electron_microscopy.manc import get_manc_loader  # noqa
from torch_em.data.sampler import MinInstanceSampler  # noqa
from torch_em.util.debug import check_loader  # noqa

CIDAS_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_manc():
    loader = get_manc_loader(
        os.path.join(CIDAS_ROOT, "manc"), patch_shape=(32, 256, 256), batch_size=1,
        download=True, sampler=MinInstanceSampler(min_num_instances=2),
    )
    check_loader(loader, 8, instance_labels=True, plt=True, save_path="./check_manc.png")


def main():
    check_manc()


if __name__ == "__main__":
    main()
