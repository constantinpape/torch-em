import os

import matplotlib
matplotlib.use("Agg")

from torch_em.data.datasets.electron_microscopy.hemibrain import get_hemibrain_loader  # noqa
from torch_em.data.sampler import MinInstanceSampler  # noqa
from torch_em.util.debug import check_loader  # noqa

CIDAS_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_hemibrain(label_choice="neurons"):
    instance_labels = label_choice == "neurons"
    loader = get_hemibrain_loader(
        os.path.join(CIDAS_ROOT, "hemibrain"), patch_shape=(32, 256, 256), batch_size=1,
        label_choice=label_choice, download=True,
        sampler=MinInstanceSampler(min_num_instances=2) if instance_labels else None,
    )
    check_loader(
        loader, 8, instance_labels=instance_labels, plt=True,
        save_path=f"./check_hemibrain_{label_choice}.png"
    )


def main():
    check_hemibrain("neurons")
    check_hemibrain("mito")
    check_hemibrain("tissue")


if __name__ == "__main__":
    main()
