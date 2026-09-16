from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_bus_uclm_loader


ROOT = "/media/anwai/ANWAI/data/bus_uclm"


def check_bus_uclm():
    loader = get_bus_uclm_loader(
        path=ROOT,
        patch_shape=(512, 512),
        batch_size=2,
        resize_inputs=False,
        download=True,
    )

    check_loader(loader, 8, rgb=True)


if __name__ == "__main__":
    check_bus_uclm()
