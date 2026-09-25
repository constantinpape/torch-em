import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.sampler import MinInstanceSampler
from torch_em.data.datasets.light_microscopy.ctc import get_ctc_segmentation_loader

sys.path.append("..")


def check_ctc_3d(dataset_name, patch_shape, annotation_type):
    from util import ROOT

    print("Checking dataset", dataset_name, "with", annotation_type, "annotations")
    loader = get_ctc_segmentation_loader(
        path=os.path.join(ROOT, "ctc"),
        dataset_name=dataset_name,
        patch_shape=patch_shape,
        batch_size=1,
        split="train",
        download=True,
        annotation_type=annotation_type,
        sampler=MinInstanceSampler(),
    )

    check_loader(loader, 8, plt=True, instance_labels=True, save_path=f"./test_{dataset_name}_{annotation_type}.png")


def main():
    # Fluo-C3DH-A549 has fully annotated volumes for some of the time points.
    check_ctc_3d("Fluo-C3DH-A549", patch_shape=(16, 256, 256), annotation_type="GT")
    # The gold truth of Fluo-N3DH-CE only annotates individual slices, so it is loaded as 2d data.
    check_ctc_3d("Fluo-N3DH-CE", patch_shape=(512, 512), annotation_type="GT")
    # The silver truth of Fluo-N3DH-CE covers full volumes and is loaded as 3d data.
    check_ctc_3d("Fluo-N3DH-CE", patch_shape=(16, 256, 256), annotation_type="ST")


if __name__ == "__main__":
    main()
