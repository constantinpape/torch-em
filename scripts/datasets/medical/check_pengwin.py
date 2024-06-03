from torch_em.data import MinInstanceSampler
from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_pengwin_loader


ROOT = "/media/anwai/ANWAI/data/pengwin"


def check_pengwin():
    loader = get_pengwin_loader(
        path=ROOT,
        patch_shape=(1, 512, 512),
        batch_size=2,
        modality="CT",
        resize_inputs=False,
        download=False,
        sampler=MinInstanceSampler(),
    )

    check_loader(loader, 8)


if __name__ == "__main__":
    check_pengwin()
