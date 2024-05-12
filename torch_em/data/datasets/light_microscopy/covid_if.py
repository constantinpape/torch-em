import os
from glob import glob

import torch_em
from . import util

COVID_IF_URL = "https://zenodo.org/record/5092850/files/covid-if-groundtruth.zip?download=1"
CHECKSUM = "d9cd6c85a19b802c771fb4ff928894b19a8fab0e0af269c49235fdac3f7a60e1"


def _download_covid_if(path, download):
    url = COVID_IF_URL
    checksum = CHECKSUM

    if os.path.exists(path):
        return

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, "covid-if.zip")
    util.download_source(zip_path, url, download, checksum)
    util.unzip(zip_path, path, True)


def get_covid_if_dataset(
    path, patch_shape, sample_range=None, target="cells", download=False,
    offsets=None, boundaries=False, binary=False, **kwargs
):
    """Dataset for the cells and nuclei in immunofluorescence.

    This dataset is from the publication https://doi.org/10.1002/bies.202000257.
    Please cite it if you use this dataset for a publication.
    """
    available_targets = ("cells", "nuclei")
    # TODO also support infected_cells
    # available_targets = ("cells", "nuclei", "infected_cells")
    assert target in available_targets, f"{target} not found in {available_targets}"

    if target == "cells":
        raw_key = "raw/serum_IgG/s0"
        label_key = "labels/cells/s0"
    elif target == "nuclei":
        raw_key = "raw/nuclei/s0"
        label_key = "labels/nuclei/s0"

    _download_covid_if(path, download)

    file_paths = sorted(glob(os.path.join(path, "*.h5")))
    if sample_range is not None:
        start, stop = sample_range
        if start is None:
            start = 0
        if stop is None:
            stop = len(file_paths)
        file_paths = [os.path.join(path, f"gt_image_{idx:03}.h5") for idx in range(start, stop)]
        assert all(os.path.exists(fp) for fp in file_paths), f"Invalid sample range {sample_range}"

    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=True, binary=binary, boundaries=boundaries, offsets=offsets
    )
    kwargs = util.update_kwargs(kwargs, "ndim", 2)

    return torch_em.default_segmentation_dataset(
        file_paths, raw_key, file_paths, label_key, patch_shape, **kwargs
    )


def get_covid_if_loader(
    path, patch_shape, batch_size, sample_range=None, target="cells", download=False,
    offsets=None, boundaries=False, binary=False, **kwargs
):
    """Dataloader for the segmentation of cells and nuclei in immunofluoroscence. See 'get_covid_if_loader' for details.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_covid_if_dataset(
        path, patch_shape, sample_range=sample_range, target=target, download=download,
        offsets=offsets, boundaries=boundaries, binary=binary, **ds_kwargs,
    )
    loader = torch_em.get_data_loader(dataset, batch_size=batch_size, **loader_kwargs)
    return loader
