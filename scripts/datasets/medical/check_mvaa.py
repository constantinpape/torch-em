import os

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.mvaa import get_mvaa_ct_loader, get_mvaa_tee_loader, get_mvaa_video_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_mvaa_ct():
    loader = get_mvaa_ct_loader(
        path=os.path.join(DATA_ROOT, "mvaa"),
        patch_shape=(32, 128, 128),
        batch_size=1,
        sampler=MinInstanceSampler(),
        download=True,
    )
    check_loader(loader, 8, plt=True, save_path="./test_mvaa_ct.png")


def check_mvaa_tee():
    loader = get_mvaa_tee_loader(
        path=os.path.join(DATA_ROOT, "mvaa"),
        patch_shape=(32, 128, 128),
        batch_size=1,
        sampler=MinInstanceSampler(),
        download=True,
    )
    check_loader(loader, 8, plt=True, save_path="./test_mvaa_tee.png")


def check_mvaa_video():
    loader = get_mvaa_video_loader(
        path=os.path.join(DATA_ROOT, "mvaa"),
        patch_shape=(512, 512),
        batch_size=1,
        sampler=MinInstanceSampler(),
        download=True,
    )
    check_loader(loader, 8, plt=True, save_path="./test_mvaa_video.png")


def main():
    check_mvaa_ct()
    check_mvaa_tee()
    check_mvaa_video()


if __name__ == "__main__":
    main()
