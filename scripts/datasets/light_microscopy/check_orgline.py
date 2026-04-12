import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets.light_microscopy.orgline import get_orgline_loader
from torch_em.data.sampler import MinForegroundSampler


sys.path.append("..")


def check_orgline_loader():
    from util import ROOT

    sampler = MinForegroundSampler(min_fraction=0.01)

    organs = "breast"
    patch_shape = (1024, 1024)
    loader = get_orgline_loader(
        path=os.path.join(ROOT, "orgline"),
        batch_size=1,
        patch_shape=patch_shape,
        organs=organs,
        split="train",
        download=True,
        sampler=sampler,
    )
    check_loader(loader, 8, instance_labels=True)


# Check images of the different organoids more systematically to assess them for training and evaluation.
# Result:
# TLDR: Can use Brain and Breast for eval (probably not for train), for the rest we either have better versions
# of the respective data or the annotation qualitty is not good enough.
# - Intestine: Look ok, but these seem to be primarily from OrgaSegment, which we have already, and might
#              already have an improved version of it. (Need to double check).
# - PDAC: From OrganoID (we have a better version of it!) and from OrganoID
#         (look weird, we have a better version of it?)
# - Brain: Only a single organoid per image, but the segmentation quality is good -> can be used.
# - Colon: Annotations are not dense, can't use.
# - Breast: Very dense images. Annotations are good overall, but in some images not everything is annotated.
#           For these, it's unclear what exactly the annotation criteria are. Ok for eval, but wouldn't train on it.
# - Stomach: Organoids are only partially annotated -> Can't really use.
def check_orgline_data():
    from glob import glob
    import h5py
    import napari
    import numpy as np
    from util import ROOT

    roots = sorted(glob(os.path.join(ROOT, "orgline", "data1", "*")) +
                   glob(os.path.join(ROOT, "orgline", "data2", "*")))

    n_images = 10
    for root in roots:
        org = os.path.basename(root)
        print("Checking organoids for", org)
        files = sorted(glob(os.path.join(root, "train", "*.h5")))
        files = np.random.choice(files, n_images, replace=False)
        for ff in files:
            with h5py.File(ff, mode="r") as f:
                image = f["image"][:]
                labels = f["masks"][:]
            v = napari.Viewer()
            v.add_image(image)
            v.add_labels(labels)
            v.title = org
            napari.run()


if __name__ == "__main__":
    # check_orgline_loader()
    check_orgline_data()
