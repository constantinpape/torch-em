import sys

from torch_em.data.datasets import get_synapseweb_hippocampus_loader
from torch_em.util.debug import check_loader

sys.path.append("..")

DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data/synapseweb_hippocampus"


def check_synapseweb_hippocampus():
    for region in ("spine", "oblique", "apical"):
        print(f"Checking region: {region}")
        loader = get_synapseweb_hippocampus_loader(
            DATA_ROOT, 1, (8, 512, 512), regions=(region,), download=True
        )
        check_loader(loader, 8, instance_labels=True, plt=True, save_path=f"check_synapseweb_hippocampus_{region}.png")


if __name__ == "__main__":
    check_synapseweb_hippocampus()
