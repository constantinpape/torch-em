import os

import torch_em

from . import util


VOLUMES = [
    "cell_1/cell_1.zarr/", ""
]


def _download_asem_dataset(path, download):
    # TEST
    target_path = "cell_6/cell_6_example.zarr/"
    if not os.path.exists(target_path):
        import quilt3
        b = quilt3.Bucket("s3://asem-project")
        b.fetch(key="datasets/cell_6/cell_6_example.zarr/", path=target_path)

    import napari
    import z5py

    with z5py.File(target_path, "r", use_zarr_format=True) as f:
        raw = f["volumes/raw_equalized_0.02"][:]
        er_labels = f["volumes/labels/er"][:]
        mito_labels = f["volumes/labels/mito"][:]
        golgi_labels = f["volumes/labels/golgi"][:]

        breakpoint()

        v = napari.Viewer()
        v.add_image(raw)
        # v.add_labels(er_labels)
        # v.add_labels(mito_labels)
        # v.add_labels(golgi_labels)
        napari.run()

        breakpoint()

    breakpoint()

    for volume_path in VOLUMES:
        b.fetch(
            key=f"datasets/{volume_path}",
            path=os.path.join(path, volume_path)
        )

    breakpoint()


def get_asem_dataset(
    path, patch_shape, ndim, download, **kwargs
):
    _download_asem_dataset(path, download)


def get_asem_loader(
    path, patch_shape, batch_size, ndim, download=False, **kwargs
):
    """TODO: description of the loader"""
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    ds = get_asem_dataset(path, patch_shape, ndim, download, **ds_kwargs)
    loader = torch_em.get_data_loader(ds, batch_size=batch_size, **loader_kwargs)
    return loader
