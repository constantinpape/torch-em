import torch_em
from . import util

ISBI_URL = "https://oc.embl.de/index.php/s/h0TkwqxU0PJDdMd/download"
CHECKSUM = "0e10fe909a1243084d91773470856993b7d40126a12e85f0f1345a7a9e512f29"


def get_isbi_dataset(
    path, patch_shape, download=False, offsets=None, boundaries=False,
    use_original_labels=False, **kwargs
):
    """Dataset for the segmentation of neurons in EM.

    This dataset is from the publication https://doi.org/10.3389/fnana.2015.00142.
    Please cite it if you use this dataset for a publication.
    """
    if not path.endswith(".h5"):
        raise ValueError("Isbi path must be a hdf5 file.")
    assert len(patch_shape) == 3
    util.download_source(path, ISBI_URL, download, CHECKSUM)
    ndim = 2 if patch_shape[0] == 1 else 3
    kwargs = util.update_kwargs(kwargs, "ndim", ndim)

    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=False, boundaries=boundaries, offsets=offsets
    )

    raw_key = "raw"
    label_key = "labels/membranes" if use_original_labels else "labels/gt_segmentation"

    return torch_em.default_segmentation_dataset(path, raw_key, path, label_key, patch_shape, **kwargs)


def get_isbi_loader(
    path, patch_shape, batch_size, download=False,
    offsets=None, boundaries=False,
    use_original_labels=False,
    **kwargs
):
    """Dataloader for the segmentation of neurons in EM. See 'get_isbi_dataset' for details."""
    ds_kwargs, loader_kwargs = util.split_kwargs(
        torch_em.default_segmentation_dataset, **kwargs
    )
    dataset = get_isbi_dataset(
        path, patch_shape, download=download,
        offsets=offsets, boundaries=boundaries, use_original_labels=use_original_labels,
        **ds_kwargs
    )
    loader = torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
    return loader
