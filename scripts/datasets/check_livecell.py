from torch_em.data.datasets import get_livecell_loader
from torch_em.util.debug import check_loader

LIVECELL_ROOT = "/home/pape/Work/data/incu_cyte/livecell"


def check_deepbacs():
    loader = get_livecell_loader(LIVECELL_ROOT, "train", (512, 512), 1)
    check_loader(loader, 15, instance_labels=True)


if __name__ == "__main__":
    check_deepbacs()
