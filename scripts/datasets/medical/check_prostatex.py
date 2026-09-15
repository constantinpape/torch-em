import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.prostatex import get_prostatex_loader


sys.path.append("..")


def check_prostatex():
    from util import ROOT

    for sequence, label_type in [
        ("t2", "lesions"), ("adc", "lesions"), ("t2", "zones"), ("t2", "zones_detailed")
    ]:
        loader = get_prostatex_loader(
            path=os.path.join(ROOT, "prostatex"),
            patch_shape=(1, 128, 128) if sequence == "adc" else (1, 384, 384),
            batch_size=1,
            ndim=2,
            sequence=sequence,
            label_type=label_type,
            sampler=MinInstanceSampler(),
            download=True,
        )

        check_loader(loader, 8, plt=True, save_path=f"./test_{sequence}_{label_type}.png")


if __name__ == "__main__":
    check_prostatex()
