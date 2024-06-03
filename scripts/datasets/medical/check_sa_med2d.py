from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_sa_med2d_loader


ROOT = "/scratch/share/cidas/cca/data/sa-med2d"


def check_sa_med2d():
    loader = get_sa_med2d_loader(
        path=ROOT,
        patch_shape=(512, 512),
        split="train",
        batch_size=2,
        resize_inputs=True,
        exclude_dataset=None,
        exclude_modality=None,
        download=False,
        num_workers=32,
    )

    check_loader(loader, 8, plt=True, save_path="./sa-med2d.png")


if __name__ == "__main__":
    check_sa_med2d()
