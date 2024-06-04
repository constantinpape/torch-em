from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_osic_pulmofib_loader


ROOT = "/media/anwai/ANWAI/data/osic_pulmofib"


def check_osic_pulmofib():
    loader = get_osic_pulmofib_loader(
        path=ROOT,
        patch_shape=(1, 512, 512),
        batch_size=2,
        resize_inputs=False,
        download=False,
    )

    check_loader(loader, 8)


def visualize_data():
    import os
    from glob import glob

    import nrrd
    import napari

    all_volume_paths = sorted(glob(os.path.join(ROOT, "nrrd_heart", "*", "*")))
    for vol_path in all_volume_paths:
        vol, header = nrrd.read(vol_path)

        v = napari.Viewer()
        v.add_image(vol.transpose(2, 0, 1))
        napari.run()


if __name__ == "__main__":
    # visualize_data()
    check_osic_pulmofib()
