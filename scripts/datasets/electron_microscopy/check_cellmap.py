import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_cellmap_loader
from torch_em.data import MinSemanticLabelForegroundSampler


sys.path.append("..")


def check_cellmap():
    from util import ROOT

    loader = get_cellmap_loader(
        path=os.path.join(ROOT, "cellmap-segmentation-challenge"),
        batch_size=2,
        patch_shape=(1, 512, 512),
        ndim=2,
        download=True,
        sampler=MinSemanticLabelForegroundSampler(semantic_ids=[3, 4, 5, 50], min_fraction=0.01),
        crops=["234"],
        voxel_size=[2.0, 2.0, 2.0],
    )
    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_cellmap()
