import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.abdomen_atlas import get_abdomen_atlas_loader


sys.path.append("..")


def check_abdomen_atlas():
    from util import ROOT

    # NOTE: The dataset is gated on HuggingFace, so the download requires an access token
    # (passed via the 'token' argument or the 'HF_TOKEN' environment variable).
    loader = get_abdomen_atlas_loader(
        path=os.path.join(ROOT, "abdomen_atlas"),
        patch_shape=(1, 512, 512),
        batch_size=1,
        ndim=2,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_abdomen_atlas()
