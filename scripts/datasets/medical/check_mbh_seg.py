from torch_em.data import MinInstanceSampler
from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_mbh_seg_loader

ROOT = "/media/anwai/ANWAI/data/mbh-seg"


def check_mbh_seg():
    loader = get_mbh_seg_loader(
        path=ROOT,
        patch_shape=(1, 512, 512),
        ndim=2,
        batch_size=2,
        resize_inputs=False,
        download=False,
        sampler=MinInstanceSampler()
    )

    check_loader(loader, 8)


def visualize_mbh_seg():
    import os
    from glob import glob
    from natsort import natsorted

    import napari
    from tukra.utils import read_image

    image_paths = natsorted(glob(os.path.join(ROOT, "images", "*.nii.gz")))
    gt_paths = natsorted(glob(os.path.join(ROOT, r"ground truths", "*.nii.gz")))
    for image_path, gt_path in zip(image_paths, gt_paths):
        image = read_image(image_path)
        gt = read_image(gt_path)

        v = napari.Viewer()
        v.add_image(image.transpose(2, 0, 1))
        v.add_labels(gt.transpose(2, 0, 1).astype("uint8"))
        napari.run()


if __name__ == "__main__":
    # visualize_mbh_seg()
    check_mbh_seg()
