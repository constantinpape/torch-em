from torch_em.data.datasets import get_covid_if_loader
from torch_em.util.debug import check_loader


def check_covid_if():
    loader = get_covid_if_loader("./data/covid-if", (512, 512), 1, download=True)
    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_covid_if()
