from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_jnuifm_loader


ROOT = "/media/anwai/ANWAI/data/jnu-ifm"


def check_jnuifm():
    loader = get_jnuifm_loader(
        path=ROOT,
        patch_shape=(256, 256),
        batch_size=2,
        download=True,
    )

    check_loader(loader, 8)


def test_jnuifm_data():
    import os
    from glob import glob
    from natsort import natsorted

    import SimpleITK as sitk

    image_paths = natsorted(glob(os.path.join(ROOT, "*", "image_mha", "*.mha")))
    gt_paths = natsorted(glob(os.path.join(ROOT, "*", "label_mha", "*.mha")))

    for image_path, gt_path in zip(image_paths, gt_paths):
        image = sitk.ReadImage(image_path)
        gt = sitk.ReadImage(gt_path)

        image = sitk.GetArrayFromImage(image)
        gt = sitk.GetArrayFromImage(gt)

        import napari
        v = napari.Viewer()
        v.add_image(image)
        v.add_labels(gt)
        napari.run()


if __name__ == "__main__":
    # test_jnuifm_data()
    check_jnuifm()
