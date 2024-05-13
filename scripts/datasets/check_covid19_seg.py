from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_covid19_seg_loader


ROOT = "/media/anwai/ANWAI/data/covid19_seg"


def check_covid19_seg():
    loader = get_covid19_seg_loader(
        path=ROOT,
        patch_shape=(1, 512, 512),
        batch_size=2,
        task="lung",
        ndim=2,
        download=True,
    )

    check_loader(loader, 8)


def test_covid19_seg():
    import os
    from glob import glob

    import nibabel as nib

    import napari

    vol_paths = sorted(glob(os.path.join(ROOT, "COVID-19-CT", "*.nii.gz")))[::-1]
    gt_paths = sorted(glob(os.path.join(ROOT, "Lung_Mask", "*.nii.gz")))[::-1]

    for vol_path, gt_path in zip(vol_paths, gt_paths):
        vol = nib.load(vol_path)
        gt = nib.load(gt_path)

        vol = vol.get_fdata()
        gt = gt.get_fdata()

        vol = vol.transpose(2, 0, 1)
        gt = gt.transpose(2, 0, 1)

        print(vol_path, vol.shape, gt_path, gt.shape)
        print()

        v = napari.Viewer()
        v.add_image(vol)
        v.add_labels(gt.astype("uint8"))
        napari.run()


if __name__ == "__main__":
    # test_covid19_seg()
    check_covid19_seg()
