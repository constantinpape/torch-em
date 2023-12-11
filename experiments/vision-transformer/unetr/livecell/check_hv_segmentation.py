import os

import imageio.v2 as imageio
import napari

LIVECELL_FOLDER = "/home/pape/Work/data/incu_cyte/livecell"


def check_hv_segmentation(image, gt):
    from torch_em.transform.label import PerObjectDistanceTransform
    from common import opencv_hovernet_instance_segmentation

    # This transform gives only directed boundary distances
    # and foreground probabilities.
    trafo = PerObjectDistanceTransform(
        distances=False,
        boundary_distances=False,
        directed_distances=True,
        foreground=True,
        min_size=10,
    )
    target = trafo(gt)
    seg = opencv_hovernet_instance_segmentation(target)

    v = napari.Viewer()
    v.add_image(image)
    v.add_image(target)
    v.add_labels(gt)
    v.add_labels(seg)
    napari.run()


def check_distance_segmentation(image, gt):
    from torch_em.transform.label import PerObjectDistanceTransform
    from torch_em.util.segmentation import watershed_from_center_and_boundary_distances

    # This transform gives distance to the centroid,
    # to the boundaries and the foreground probabilities
    trafo = PerObjectDistanceTransform(
        distances=True,
        boundary_distances=True,
        directed_distances=False,
        foreground=True,
        min_size=10,
    )
    target = trafo(gt)

    # run the segmentation
    fg, cdist, bdist = target
    seg = watershed_from_center_and_boundary_distances(
        cdist, bdist, fg, min_size=50,
    )

    # visualize it
    v = napari.Viewer()
    v.add_image(image)
    v.add_image(target)
    v.add_labels(gt)
    v.add_labels(seg)
    napari.run()


def main():
    # load image and ground-truth from LiveCELL
    fname = "A172_Phase_A7_1_01d00h00m_1.tif"
    image_path = os.path.join(LIVECELL_FOLDER, "images/livecell_train_val_images", fname)
    image = imageio.imread(image_path)

    label_path = os.path.join(LIVECELL_FOLDER, "annotations/livecell_train_val_images/A172", fname)
    gt = imageio.imread(label_path)

    # Check the hovernet instance segmentation on GT.
    check_hv_segmentation(image, gt)

    # Check the new distance based segmentation on GT.
    check_distance_segmentation(image, gt)


if __name__ == "__main__":
    main()
