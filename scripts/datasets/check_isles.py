from torch_em.util.debug import check_loader
from torch_em.data import MinForegroundSampler
from torch_em.data.datasets.medical import get_isles_loader


ROOT = "/media/anwai/ANWAI/data/isles"


def check_isles():
    loader = get_isles_loader(
        path=ROOT,
        patch_shape=(1, 112, 112),
        batch_size=2,
        ndim=2,
        modality=None,
        download=False,
        sampler=MinForegroundSampler(min_fraction=0.001),
    )
    check_loader(loader, 8)


def test_isles_images():
    import os
    from glob import glob

    import napari
    import nibabel as nib

    patient_dirs = glob(os.path.join(ROOT, "ISLES-2022", "sub-*"))

    for patient_dir in patient_dirs:
        patient_id = os.path.split(patient_dir)[-1]

        gt_path = glob(os.path.join(ROOT, "ISLES-2022", "derivatives", patient_id, "**", "*.nii.gz"), recursive=True)[0]
        gt = nib.load(gt_path)
        gt = gt.get_fdata()
        print(gt.shape)

        # v = napari.Viewer()
        # v.add_labels(gt.transpose(2, 0, 1).astype("uint8"))

        all_volume_paths = glob(os.path.join(patient_dir, "**", "dwi", "*.nii.gz"), recursive=True)
        for vol_path in all_volume_paths:
            # print(vol_path)
            vol = nib.load(vol_path)
            vol = vol.get_fdata()
            # v.add_image(vol.transpose(2, 0, 1))
            print(vol.shape)

        # napari.run()

        print()


if __name__ == "__main__":
    # test_isles_images()
    check_isles()
