from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_msd_loader


MSD_ROOT = "/media/anwai/ANWAI/data/msd"


def check_msd():
    loader = get_msd_loader(
        path=MSD_ROOT,
        patch_shape=(1, 512, 512),
        batch_size=2,
        ndim=2,
        download=True,
        task_names="braintumour",
    )
    print(f"Length of the loader: {len(loader)}")
    check_loader(loader, 8)


if __name__ == "__main__":
    check_msd()
