import os

import matplotlib
matplotlib.use("Agg")

from torch_em.data.datasets.electron_microscopy.zebrafinch import get_zebrafinch_loader  # noqa
from torch_em.data.sampler import MinInstanceSampler  # noqa
from torch_em.util.debug import check_loader  # noqa

CIDAS_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"

# Sub-region at mip=0 (10x10x25 nm): ~512 z-planes, 5120x5120 xy.
J0251_BBOX = (0, 51200, 0, 51200, 0, 12800)
# Sub-region at mip=0 (10x10x20 nm): ~640 z-planes, 5120x5120 xy.
J0126_BBOX = (0, 51200, 0, 51200, 0, 12800)

DATASET_BBOXES = {"j0251": J0251_BBOX, "j0126": J0126_BBOX}


def check_zebrafinch(dataset="j0251", label_choice="neurons"):
    loader = get_zebrafinch_loader(
        os.path.join(CIDAS_ROOT, "zebrafinch"), batch_size=1, patch_shape=(32, 256, 256),
        download=True, dataset=dataset, bounding_box=DATASET_BBOXES[dataset], label_choice=label_choice,
        sampler=MinInstanceSampler(min_num_instances=2),
    )
    check_loader(
        loader, 8, instance_labels=True, plt=True,
        save_path=f"./check_zebrafinch_{dataset}_{label_choice}.png"
    )


def main():
    check_zebrafinch("j0251", "neurons")
    check_zebrafinch("j0251", "er")
    check_zebrafinch("j0126", "neurons")


if __name__ == "__main__":
    main()
