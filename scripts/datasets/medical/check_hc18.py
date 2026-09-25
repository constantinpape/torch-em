import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.hc18 import get_hc18_loader

sys.path.append("..")


def check_hc18():
    from util import ROOT

    loader = get_hc18_loader(
        path=os.path.join(ROOT, "hc18"),
        patch_shape=(512, 512),
        batch_size=2,
        resize_inputs=True,
        download=True,
    )

    save_path = os.environ.get("HC18_CHECK_SAVE_PATH")
    check_loader(loader, 8, plt=save_path is not None, save_path=save_path)


if __name__ == "__main__":
    check_hc18()
