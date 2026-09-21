import os

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.semi_teethseg import get_semi_teethseg_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_semi_teethseg_2d():
    loader = get_semi_teethseg_loader(
        path=os.path.join(DATA_ROOT, "semi_teethseg"),
        batch_size=2,
        patch_shape=(512, 512),
        split="adult",
        modality="2d",
        resize_inputs=True,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test_2d.png")


def check_semi_teethseg_3d():
    for subset in ["roi", "integrity"]:
        loader = get_semi_teethseg_loader(
            path=os.path.join(DATA_ROOT, "semi_teethseg"),
            batch_size=1,
            patch_shape=(32, 512, 512),
            modality="3d",
            subset=subset,
            download=True,
        )

        check_loader(loader, 4, plt=True, save_path=f"./test_3d_{subset}.png")


if __name__ == "__main__":
    check_semi_teethseg_2d()
    check_semi_teethseg_3d()
