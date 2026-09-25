import os

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.maternal_fetal_us_video import get_maternal_fetal_us_video_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_maternal_fetal_us_video():
    loader = get_maternal_fetal_us_video_loader(
        path=os.path.join(DATA_ROOT, "maternal_fetal_us_video"),
        batch_size=2,
        patch_shape=(512, 512),
        split="all",
        resize_inputs=True,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test_maternal_fetal_us_video.png")


if __name__ == "__main__":
    check_maternal_fetal_us_video()
