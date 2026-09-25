import os

import matplotlib
matplotlib.use("Agg")

from torch_em.data.datasets.electron_microscopy.parlakgul_liver import get_parlakgul_liver_loader  # noqa
from torch_em.util.debug import check_loader  # noqa

CIDAS_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"

# Small bbox for check: 512x512x16 voxels at 8nm = ~4x4x0.13 um
# Central region, 64 z-slices, 2048x2048 xy
CHECK_BBOX = (4000, 6048, 2000, 4048, 2000, 2064)


def check_parlakgul_liver(label_choice="mito"):
    loader = get_parlakgul_liver_loader(
        os.path.join(CIDAS_ROOT, "parlakgul_liver"), patch_shape=(32, 512, 512), batch_size=1,
        bounding_boxes=[CHECK_BBOX], sample="6461", label_choice=label_choice, download=True,
    )
    check_loader(loader, 8, plt=True, save_path=f"./check_parlakgul_{label_choice}.png")


def main():
    check_parlakgul_liver("mito")


if __name__ == "__main__":
    main()
