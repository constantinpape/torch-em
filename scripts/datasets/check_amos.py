from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical import get_amos_loader

ROOT = "/media/anwai/ANWAI/data/amos"


def check_amos():
    loader = get_amos_loader(
        path=ROOT,
        split="train",
        patch_shape=(1, 512, 512),
        modality="mri",
        ndim=2,
        batch_size=2,
        download=False,
        sampler=MinInstanceSampler(min_num_instances=3),
    )
    check_loader(loader, 8)


def test_amos_images():
    import os
    from glob import glob

    import numpy as np
    import nibabel as nib

    image_paths = sorted(glob(os.path.join(ROOT, "amos22", "imagesTr", "*.nii.gz")))
    gt_paths = sorted(glob(os.path.join(ROOT, "amos22", "labelsTr", "*.nii.gz")))

    for image_path, gt_path in zip(image_paths, gt_paths):
        image = nib.load(image_path)
        gt = nib.load(gt_path)

        image = image.get_fdata()
        gt = gt.get_fdata()

        # import napari
        # v = napari.Viewer()
        # v.add_image(image.transpose(2, 0, 1))
        # v.add_labels(gt.astype("uint8").transpose(2, 0, 1))
        # napari.run()

        print(len(np.unique(gt)))


if __name__ == "__main__":
    # test_amos_images()
    check_amos()
