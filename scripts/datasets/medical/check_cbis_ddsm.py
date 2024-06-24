from torch_em.data import MinInstanceSampler
from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_cbis_ddsm_loader


ROOT = "/media/anwai/ANWAI/data/cbis_ddsm"


def check_cbis_ddsm():
    loader = get_cbis_ddsm_loader(
        path=ROOT,
        patch_shape=(512, 512),
        batch_size=2,
        split="Train",
        task=None,
        tumour_type=None,
        resize_inputs=True,
        sampler=MinInstanceSampler()
    )
    check_loader(loader, 8)


if __name__ == "__main__":
    check_cbis_ddsm()
