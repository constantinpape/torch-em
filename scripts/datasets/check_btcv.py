from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_btcv_loader
from torch_em.data import MinTwoInstanceSampler

BTCV_ROOT = "/scratch/usr/nimanwai/data/btcv/"


def check_btcv():
    loader = get_btcv_loader(
        path=BTCV_ROOT,
        patch_shape=(1, 512, 512),
        batch_size=2,
        ndim=2,
        sampler=MinTwoInstanceSampler(),
        anatomy=None,
        organs=["aorta", "bladder", "uterus"]
    )
    print(f"Length of the loader: {len(loader)}")
    check_loader(loader, 8, plt=True, save_path="btcv.png")


if __name__ == "__main__":
    check_btcv()
