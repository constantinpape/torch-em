from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical import get_toothfairy_loader


ROOT = "/scratch/share/cidas/cca/data/toothfairy/"


def check_toothfairy():
    loader = get_toothfairy_loader(
        path=ROOT,
        patch_shape=(1, 512, 512),
        ndim=2,
        batch_size=2,
        sampler=MinInstanceSampler()
    )

    check_loader(loader, 8, plt=True, save_path="./toothfairy.png")


check_toothfairy()
