from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_covid_qu_ex_loader


ROOT = "/media/anwai/ANWAI/data/covid-qu-ex"


def check_covid_qu_ex():
    loader = get_covid_qu_ex_loader(
        path=ROOT,
        patch_shape=(256, 256),
        batch_size=2,
        split="train",  # train / val / test
        task="lung",  # lung / infection
        patient_type="covid19",  # covid19 / non-covid / normal
        segmentation_mask="lung",  # lung / infection
        download=False,
    )

    check_loader(loader, 8)


if __name__ == "__main__":
    check_covid_qu_ex()
