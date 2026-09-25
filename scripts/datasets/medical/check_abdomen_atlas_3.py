import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data import MinSemanticLabelForegroundSampler
from torch_em.data.datasets.medical.abdomen_atlas_3 import get_abdomen_atlas_3_loader, CLASS_IDS, LESION_NAMES


sys.path.append("..")


def check_abdomen_atlas_3():
    from util import ROOT

    # NOTE: The dataset is not gated on HuggingFace, so no access token is required. It is very large
    # (~586 GB in total), so 'max_cases' is used here to only download the shard(s) covering a few cases.
    lesion_ids = [CLASS_IDS[name] for name in LESION_NAMES]
    sampler = MinSemanticLabelForegroundSampler(semantic_ids=lesion_ids, min_fraction=0.0005)

    loader = get_abdomen_atlas_3_loader(
        path=os.path.join(ROOT, "abdomen_atlas_3"),
        patch_shape=(1, 512, 512),
        batch_size=1,
        max_cases=5,
        ndim=2,
        sampler=sampler,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_abdomen_atlas_3()
