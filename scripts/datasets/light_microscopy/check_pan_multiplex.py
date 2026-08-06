import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_pan_multiplex_loader


sys.path.append("..")


def check_pan_multiplex():
    from util import ROOT

    # 'both' shows the nuclei and the membrane composite as two channels in the viewer.
    for subset in ["mibi_decidua", "codex_colon", "vectra_colon", "vectra_pancreas", "mibi_breast"]:
        print(f"Checking {subset} ...")
        loader = get_pan_multiplex_loader(
            path=os.path.join(ROOT, "pan_multiplex"),
            patch_shape=(512, 512),
            batch_size=2,
            subset=subset,
            split="train",
            raw_channel="both",
            download=True,
            shuffle=True,
        )

        check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_pan_multiplex()
