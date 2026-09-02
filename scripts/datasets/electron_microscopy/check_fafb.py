import os

import matplotlib
matplotlib.use("Agg")

from torch_em.data.datasets.electron_microscopy.fafb import get_fafb_loader  # noqa
from torch_em.data.sampler import MinInstanceSampler  # noqa
from torch_em.util.debug import check_loader  # noqa

CIDAS_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"

# Smaller bbox for check: 512x512x205 voxels = ~8x8x8 um isotropic at 16x16x40nm.
CHECK_BBOX = (31000, 31512, 14500, 15012, 3200, 3405)


def check_fafb():
    loader = get_fafb_loader(
        os.path.join(CIDAS_ROOT, "fafb"), patch_shape=(64, 128, 128), batch_size=1,
        bounding_boxes=[CHECK_BBOX], download=True,
        sampler=MinInstanceSampler(min_num_instances=2),
    )
    check_loader(loader, 8, instance_labels=True, plt=True, save_path="./check_fafb.png")


def main():
    check_fafb()


if __name__ == "__main__":
    main()
