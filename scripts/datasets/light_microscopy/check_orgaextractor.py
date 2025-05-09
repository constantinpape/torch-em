import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_orgaextractor_loader


sys.path.append("..")


def check_orgaextractor():
    # from util import ROOT
    ROOT = "/media/anwai/ANWAI/data"

    loader = get_orgaextractor_loader(
        path=os.path.join(ROOT, "orgaextractor"),
        batch_size=2,
        patch_shape=(512, 512),
        split="train",
        download=True,
    )

    check_loader(loader, 8, instance_labels=True)


def test_stuff(fraction=0.1):
    import torch_em
    from torch_em.data.datasets import get_livecell_dataset, livecell

    datasets = []
    for cell_type in livecell.CELL_TYPES:

        image_paths, _ = livecell.get_livecell_paths(
            path=...,
            split="train",
            cell_types=[cell_type]
        )

        datasets.append(
            get_livecell_dataset(
                path=...,
                split="train",
                patch_shape=...,
                cell_types=[cell_type],
                n_samples=len(image_paths) * fraction,
            )
        )

    from torch_em.data import ConcatDataset
    final_dataset = ConcatDataset(datasets)

    final_loader = torch_em.get_data_loader(final_dataset, batch_size=2)


if __name__ == "__main__":
    check_orgaextractor()
