import os

import torch_em
from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.topaneu import get_topaneu_dataset


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_topaneu():
    # The aneurysm location masks are tiny compared to the volume, so a large 3d patch is used to reliably
    # find a foreground patch (as also done for the ISLES datasets).
    dataset = get_topaneu_dataset(
        path=os.path.join(DATA_ROOT, "topaneu"),
        patch_shape=(64, 384, 384),
        label_choice="location",
        modality=None,
        sampler=MinInstanceSampler(),
        ndim=3,
        download=False,
    )
    for ds in dataset.datasets:
        ds.max_sampling_attempts = 2000

    loader = torch_em.get_data_loader(dataset, batch_size=1)

    check_loader(loader, 8, plt=True, save_path="./test_topaneu.png")


if __name__ == "__main__":
    check_topaneu()
