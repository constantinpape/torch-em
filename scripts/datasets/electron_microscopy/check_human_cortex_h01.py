import os

import matplotlib
matplotlib.use("Agg")

from torch_em.data.datasets.electron_microscopy.human_cortex_h01 import get_human_cortex_h01_loader  # noqa
from torch_em.data.sampler import MinInstanceSampler  # noqa
from torch_em.util.debug import check_loader  # noqa

CIDAS_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"

# Smaller bbox for check: 512x512x128 voxels = ~4x4x4.2 um isotropic at 8x8x33nm.
CHECK_BBOX = (300000, 300512, 200000, 200512, 3000, 3128)


def check_human_cortex_h01():
    loader = get_human_cortex_h01_loader(
        os.path.join(CIDAS_ROOT, "h01"), patch_shape=(64, 128, 128), batch_size=1,
        bounding_boxes=[CHECK_BBOX], download=True,
        sampler=MinInstanceSampler(min_num_instances=2),
    )
    check_loader(loader, 8, instance_labels=True, plt=True, save_path="./check_human_cortex_h01.png")


def main():
    check_human_cortex_h01()


if __name__ == "__main__":
    main()
