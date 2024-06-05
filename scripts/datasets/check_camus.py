from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_camus_loader


ROOT = "/media/anwai/ANWAI/data/camus"


def check_camus():
    loader = get_camus_loader(
        path=ROOT,
        patch_shape=(1, 256, 256),
        batch_size=2,
        chamber=2,
        resize_inputs=True,
        download=False,
    )
    check_loader(loader, 8)


def test_camus_images():
    import os
    from glob import glob

    import napari
    import nibabel as nib

    all_patient_dir = glob(os.path.join(ROOT, "database_nifti", "patient*"))

    # v = napari.Viewer()

    for per_patient_dir in all_patient_dir:
        all_volume_paths = sorted(glob(os.path.join(per_patient_dir, "*_4CH_*.nii.gz")))
        for vol_path in all_volume_paths:
            vol = nib.load(vol_path)
            vol = vol.get_fdata()

            if vol.ndim == 2:
                print(vol.shape)
                # v.add_image(vol)

        print()

        # napari.run()

        # breakpoint()


if __name__ == "__main__":
    # test_camus_images()
    check_camus()
