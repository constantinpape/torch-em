import torch_em.data.datasets.platynereis as platy
from torch_em.util.debug import check_loader


def check_platynereis():

    # check nucleus lader
    # loader = platy.get_platynereis_nuclei_loader("./data/platy", (8, 512, 512), 1, download=True)
    # check_loader(loader, 8, instance_labels=True)

    # check cell loader
    # loader = platy.get_platynereis_cell_loader("./data/platy", (8, 512, 512), 1, download=True)
    # check_loader(loader, 8, instance_labels=True)

    # check cilia loader
    # loader = platy.get_platynereis_cilia_loader("./data/platy", "train", (8, 512, 512), 1, download=True)
    # check_loader(loader, 8, instance_labels=True)

    # check cuticle loader
    loader = platy.get_platynereis_cuticle_loader("./data/platy", (8, 512, 512), 1, download=True)
    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_platynereis()
