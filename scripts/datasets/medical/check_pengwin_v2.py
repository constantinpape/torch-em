import os

import torch_em
from torch_em.data import MinInstanceSampler
from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.pengwin import get_pengwin_paths


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_pengwin_v2():
    image_paths, gt_paths = get_pengwin_paths(
        path=os.path.join(DATA_ROOT, "pengwin26"), modality="CT", download=True, version="v2",
    )

    # Use a meaningful subset of the 340 labeled cases to keep memory / runtime bounded for this check.
    n_subset = 5
    image_paths, gt_paths = image_paths[:n_subset], gt_paths[:n_subset]

    dataset = torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=gt_paths,
        label_key=None,
        patch_shape=(1, 512, 512),
        is_seg_dataset=True,
        sampler=MinInstanceSampler(),
    )
    loader = torch_em.get_data_loader(dataset, batch_size=1, num_workers=0)

    check_loader(loader, 8, instance_labels=True, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_pengwin_v2()
