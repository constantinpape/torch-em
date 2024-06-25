import os
import sys

from torch_em.data.datasets import get_covid_if_loader
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_covid_if():
    from util import ROOT

    loader = get_covid_if_loader(os.path.join(ROOT, "covid-if"), (512, 512), 1, download=True)
    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_covid_if()
