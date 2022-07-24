import argparse
import os
from glob import glob

import torch_em
import torch_em.shallow2deep as shallow2deep
from torch_em.data.datasets.mouse_embryo import _require_embryo_data
from torch_em.data.datasets.plantseg import _require_plantseg_data
from torch_em.shallow2deep.prepare_shallow2deep import _prepare_shallow2deep

DATA_ROOT = "/scratch/pape/s2d-lm-boundaries"


# check the rf loader to see if samplers for the complicated 3d datasets work
# - mouse-embryo:
#   - 2d: default sampler is fine
#   - 3d: TODO
# - ovules:
#   - 2d: default sampler is fine
#   - 3d: TODO
# - root:
#   - 2d: default sampler is fine
#   - 3d: TODO
#


def require_ds(dataset):
    os.makedirs(DATA_ROOT, exist_ok=True)
    data_path = os.path.join(DATA_ROOT, dataset)
    if dataset == "mouse-embryo":
        _require_embryo_data(data_path, True)
        paths = glob(os.path.join(data_path, "Membrane", "train", "*.h5"))
        raw_key, label_key = "raw", "label"
    elif dataset == "ovules":
        _require_plantseg_data(data_path, True, "ovules", "train")
        paths = glob(os.path.join(data_path, "ovules_train", "*.h5"))
        raw_key, label_key = "raw", "label"
    elif dataset == "root":
        _require_plantseg_data(data_path, True, "root", "train")
        paths = glob(os.path.join(data_path, "root_train", "*.h5"))
        raw_key, label_key = "raw", "label"
    return paths, raw_key, label_key


def check_rf_loader(dataset, ndim):
    assert dataset in ("mouse-embryo", "ovules", "root")
    paths, raw_key, label_key = require_ds(dataset)
    n_images = 16
    if ndim == 2:
        patch_shape_min = [1, 248, 248]
        patch_shape_max = [1, 256, 256]
    else:
        pass  # TODO
    # TODO sampler
    raw_transform = torch_em.transform.raw.normalize
    label_transform = shallow2deep.BoundaryTransform(ndim=ndim)
    ds, _ = _prepare_shallow2deep(
        paths, raw_key, paths, label_key,
        patch_shape_min, patch_shape_max,
        n_forests=n_images, ndim=ndim,
        raw_transform=raw_transform, label_transform=label_transform,
        rois=None, filter_config=None, sampler=None,
        is_seg_dataset=True,
    )

    print("Start viewer")
    import napari
    for i in range(len(ds)):
        x, y = ds[i]
        v = napari.Viewer()
        v.add_image(x.numpy().squeeze(), name="data")
        v.add_image(y.numpy().squeeze(), name="target")
        napari.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", required=True)
    parser.add_argument("-n", "--ndim", default=2, type=int)
    args = parser.parse_args()
    check_rf_loader(args.dataset, args.ndim)
