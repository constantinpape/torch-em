from torch_em.util.debug import check_loader
from torch_em.data.datasets.light_microscopy import get_organoidnet_loader


ROOT = "../data"


def check_organoidnet():
    loader = get_organoidnet_loader(
        path=ROOT,
        split="Training",
        patch_shape=(512, 512),
        batch_size=1,
        download=True,
    )
    check_loader(loader, 8, instance_labels=True)


def _check_split(split):
    import os
    from glob import glob
    import imageio.v3 as imageio
    import napari

    root = os.path.join(ROOT, split)
    # image_files = sorted(glob(os.path.join(root, "Images", "Mouse_2*.tif")))
    # mask_files = sorted(glob(os.path.join(root, "Masks", "Mouse_2*.tif")))
    image_files = sorted(glob(os.path.join(root, "Images", "*.tif")))
    mask_files = sorted(glob(os.path.join(root, "Masks", "*.tif")))

    for imf, maskf in zip(image_files, mask_files):
        image = imageio.imread(imf)
        masks = imageio.imread(maskf)

        v = napari.Viewer()
        v.add_image(image)
        v.add_labels(masks)
        v.title = os.path.basename(imf)
        napari.run()


def exhaustive_check():
    # Data format and duplication:
    # In the Train and Val split each image is divided into 6 (0-5) patches,
    # but 1-2, and 4-5 are identical except for some small shift.
    # The other patches also share partial overlaps.

    # Training split: for "Human": overlaps are predominantly not marked as individual IDs.
    # Exceptions:
    # Human_2_004h_patch1.tif
    # Human_4_008h_patch3.tif
    # For Mouse: happens much more frequently
    # Mouse_1_000h_patch1.tif
    # Mouse_1_000h_patch3.tif
    # Mouse_1_002h_patch1.tif
    # ... (Stopped counting after this, it's the case for more data from Mouse_1)
    # Mouse_2 also has the problem for many images. Though it's not a problem for all of them.
    # _check_split("Training")

    # Similar to Training: The effect is bigger in Mouse compared to Human
    # Also: Data leakage! The images in Val and Train seem to be the same,
    # e.g. "Mouse_2_000h_patch1" is the same in Train and Val.
    # Same for some other patches.
    # _check_split("Validation")

    # Bigger images, no data leakage.
    # The human samples are quite sparsely covered with organoids, the mouse samples are covered very densely.
    # Again, human samples are ok, mouse samples have a lot of overlapping annotations.
    _check_split("Test")


def main():
    # check_organoidnet()
    exhaustive_check()


if __name__ == "__main__":
    main()
