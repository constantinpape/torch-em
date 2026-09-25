import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_covid_qu_ex_loader


sys.path.append("..")


def check_covid_qu_ex():
    from util import ROOT

    loader = get_covid_qu_ex_loader(
        path=os.path.join(ROOT, "covid_qu_ex"),
        patch_shape=(256, 256),
        batch_size=2,
        split="train",  # train / val / test
        task="lung",  # lung / infection
        patient_type="covid19",  # covid19 / non-covid / normal
        segmentation_mask="lung",  # lung / infection
        download=True,
    )

    check_loader(loader, 8)


if __name__ == "__main__":
    check_covid_qu_ex()
