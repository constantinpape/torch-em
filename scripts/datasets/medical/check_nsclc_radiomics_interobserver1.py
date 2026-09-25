import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.nsclc_radiomics_interobserver1 import get_nsclc_radiomics_interobserver1_loader

sys.path.append("..")


def check_nsclc_radiomics_interobserver1():
    from util import ROOT

    loader = get_nsclc_radiomics_interobserver1_loader(
        path=os.path.join(ROOT, "nsclc_radiomics_interobserver1"),
        patch_shape=(1, 512, 512),
        batch_size=1,
        annotator=1,
        annotation_type="manual",
        ndim=2,
        sampler=MinInstanceSampler(min_num_instances=2),
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_nsclc_radiomics_interobserver1()
