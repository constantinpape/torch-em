import os

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.npc_mri import get_npc_mri_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_npc_mri():
    loader = get_npc_mri_loader(
        path=os.path.join(DATA_ROOT, "npc_mri"),
        batch_size=1,
        patch_shape=(1, 512, 512),
        modality="CE-T1",
        ndim=2,
        sampler=MinInstanceSampler(),
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test_npc_mri.png")


if __name__ == "__main__":
    check_npc_mri()
