import os
import sys

from torch_em.data.datasets import get_optic_lobe_loader
from torch_em.data.sampler import MinInstanceSampler
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_optic_lobe():
    from util import ROOT

    loader = get_optic_lobe_loader(
        os.path.join(ROOT, "optic_lobe"), patch_shape=(32, 256, 256), batch_size=1,
        download=True, sampler=MinInstanceSampler(min_num_instances=2),
    )
    check_loader(loader, 4, instance_labels=True)


if __name__ == "__main__":
    check_optic_lobe()
