import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.sampler import MinInstanceSampler
from torch_em.data.datasets.light_microscopy.ctc import get_ctc_segmentation_loader, CTC_2D_DATASETS

sys.path.append("..")


# Some of the datasets have partial sparse labels:
# - Fluo-N2DH-GOWT1
# - Fluo-N2DL-HeLa
# Maybe depends on the split?!
# The 3d datasets are checked in check_ctc_3d.py.
def check_ctc_segmentation(split):
    from util import ROOT, USE_NAPARI

    data_root = os.path.join(ROOT, "ctc")
    ctc_dataset_names = CTC_2D_DATASETS
    for name in ctc_dataset_names:
        print("Checking dataset", name)
        loader = get_ctc_segmentation_loader(
            path=data_root,
            dataset_name=name,
            patch_shape=(1, 512, 512),
            batch_size=1,
            download=True,
            split=split,
            sampler=MinInstanceSampler()
        )
        check_loader(loader, 8, plt=not USE_NAPARI, instance_labels=True)


if __name__ == "__main__":
    check_ctc_segmentation("train")
