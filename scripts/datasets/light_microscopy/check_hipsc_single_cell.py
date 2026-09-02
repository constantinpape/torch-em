import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_hipsc_single_cell_loader


sys.path.append("..")


def check_hipsc_single_cell():
    from util import ROOT

    path = os.path.join(ROOT, "hipsc_single_cell")
    settings = [
        ("cell", "structure", (32, 128, 128), False),
        ("fov", "nucleus", (16, 128, 128), True),
    ]
    for sample_type, target, patch_shape, instance_labels in settings:
        loader = get_hipsc_single_cell_loader(
            path=path,
            batch_size=1,
            patch_shape=patch_shape,
            structure_names=["TOMM20"],
            sample_type=sample_type,
            target=target,
            n_samples=5,
            download=True,
        )
        check_loader(loader, 4, instance_labels=instance_labels)


if __name__ == "__main__":
    check_hipsc_single_cell()
