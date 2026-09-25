import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_myofuse_loader


sys.path.append("..")


def check_myofuse():
    from util import ROOT

    for subset in ["human", "mouse"]:
        for label_choice in ["semantic", "instances"]:
            loader = get_myofuse_loader(
                path=os.path.join(ROOT, "myofuse"),
                patch_shape=(512, 512),
                batch_size=2,
                subset=subset,
                label_choice=label_choice,
                download=True,
                shuffle=True,
            )

            check_loader(loader, 8, instance_labels=(label_choice == "instances"))


if __name__ == "__main__":
    check_myofuse()
