import os
import sys

from torch_em.data.datasets.light_microscopy.micro_bench import get_micro_bench_loader
from torch_em.util.debug import check_loader


sys.path.append("..")


def check_micro_bench():
    from util import ROOT

    for source in ("burgess", "cellcognition", "opencell", "wu"):
        loader = get_micro_bench_loader(
            path=os.path.join(ROOT, "micro_bench"),
            batch_size=1,
            patch_shape=(224, 224),
            source=source,
            download=True,
        )
        check_loader(loader, 4, instance_labels=True, rgb=True)


def main():
    check_micro_bench()


if __name__ == "__main__":
    main()
