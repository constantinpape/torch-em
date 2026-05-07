import os
import sys

from torch_em.data import MinInstanceSampler
from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_acouslic_ai_loader


sys.path.append("..")


def check_acouslic_ai():
    from util import ROOT

    loader = get_acouslic_ai_loader(
        path=os.path.join(ROOT, "acouslic_ai"),
        patch_shape=(1, 512, 512),
        ndim=2,
        batch_size=1,
        resize_inputs=False,
        download=True,
        sampler=MinInstanceSampler(),
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_acouslic_ai()
