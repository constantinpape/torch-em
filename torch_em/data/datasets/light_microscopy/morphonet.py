"""The MorphoNet dataset contains 3D segmentation annotations for five organisms
from the MorphoNet 2.0 publication.

The dataset provides 3D instance segmentation labels for cell and nucleus segmentation
across five model organisms imaged with confocal and light-sheet microscopy:
- Patiria miniata (starfish embryo, confocal, membrane/nuclei) from https://doi.org/10.1242/dev.202362.
- Tribolium castaneum (beetle embryo, light-sheet, nuclei) from https://doi.org/10.1038/s41592-023-01879-y.
- Arabidopsis thaliana (plant shoot apical meristem, confocal, membranes) from https://doi.org/10.1073/pnas.1616768113.
- Caenorhabditis elegans (nematode embryo, confocal, nuclei) from https://doi.org/10.1038/s41592-023-01879-y.
- Phallusia mammillata (ascidian embryo, light-sheet, membranes) from https://doi.org/10.1126/science.aar5663.

The dataset is located at https://doi.org/10.6084/m9.figshare.30529745.v2.
This dataset is from the publication https://doi.org/10.7554/eLife.106227.2.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from typing import Union, Tuple, List, Optional

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = {
    "patiria_miniata": "https://ndownloader.figshare.com/files/59296676",
    "tribolium_castaneum": "https://ndownloader.figshare.com/files/59296685",
    "arabidopsis_thaliana": "https://ndownloader.figshare.com/files/59296700",
    "caenorhabditis_elegans": "https://ndownloader.figshare.com/files/59296703",
    "phallusia_mammillata": "https://ndownloader.figshare.com/files/59296712",
}

DIR_NAMES = {
    "patiria_miniata": "Patiria miniata",
    "tribolium_castaneum": "Tribolium castaneum",
    "arabidopsis_thaliana": "Arabidopsis thaliana",
    "caenorhabditis_elegans": "Caenorhabditis elegans",
    "phallusia_mammillata": "Phallusia mammillata",
}

ORGANISMS = list(URLS.keys())


def _get_tif_files(directory):
    """Get all TIF/TIFF files from a directory."""
    files = glob(os.path.join(directory, "*.tif")) + glob(os.path.join(directory, "*.tiff"))
    # Exclude macOS metadata files.
    files = [f for f in files if not os.path.basename(f).startswith(".")]
    return files


def _match_raw_seg_files(raw_dir, seg_dir, organism):
    """Match RAW and SEG files for a given organism."""
    from natsort import natsorted

    raw_files = natsorted(_get_tif_files(raw_dir))
    seg_files = natsorted(_get_tif_files(seg_dir))

    # For Tribolium, filter out the empty channel 0 from RAW (only ch1 has nuclei).
    if organism == "tribolium_castaneum":
        raw_files = [f for f in raw_files if "ch0" not in os.path.basename(f)]

    assert len(raw_files) > 0, f"No RAW files found in {raw_dir}"
    assert len(seg_files) > 0, f"No SEG files found in {seg_dir}"
    assert len(raw_files) == len(seg_files), (
        f"Mismatch for {organism}: {len(raw_files)} RAW files vs {len(seg_files)} SEG files"
    )

    return list(zip(raw_files, seg_files))


def _create_h5_data(path, organism):
    """Create h5 files with raw images and instance segmentation labels."""
    import h5py
    import imageio.v3 as imageio
    from tqdm import tqdm

    h5_dir = os.path.join(path, "h5_data", organism)
    os.makedirs(h5_dir, exist_ok=True)

    org_dir = os.path.join(path, DIR_NAMES[organism])
    raw_dir = os.path.join(org_dir, "published", "RAW")
    seg_dir = os.path.join(org_dir, "published", "SEG")

    pairs = _match_raw_seg_files(raw_dir, seg_dir, organism)

    for i, (raw_path, seg_path) in enumerate(tqdm(pairs, desc=f"Creating h5 for {organism}")):
        h5_path = os.path.join(h5_dir, f"t{i:04d}.h5")

        if os.path.exists(h5_path):
            continue

        raw = imageio.imread(raw_path)
        seg = imageio.imread(seg_path)

        with h5py.File(h5_path, "w") as f:
            f.create_dataset("raw", data=raw, compression="gzip")
            f.create_dataset("labels", data=seg.astype("int64"), compression="gzip")

    return h5_dir


def get_morphonet_data(
    path: Union[os.PathLike, str],
    organism: Optional[Union[str, List[str]]] = None,
    download: bool = False,
) -> str:
    """Download the MorphoNet dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        organism: The organism(s) to download data for. If None, all organisms will be downloaded.
            Available organisms: patiria_miniata, tribolium_castaneum, arabidopsis_thaliana,
            caenorhabditis_elegans, phallusia_mammillata.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the directory with the data.
    """
    if organism is None:
        organism = ORGANISMS
    elif isinstance(organism, str):
        organism = [organism]

    for org in organism:
        assert org in ORGANISMS, f"'{org}' is not a valid organism. Choose from {ORGANISMS}."

        org_dir = os.path.join(path, DIR_NAMES[org])
        if os.path.exists(org_dir):
            continue

        os.makedirs(path, exist_ok=True)
        zip_path = os.path.join(path, f"{org}.zip")
        util.download_source(path=zip_path, url=URLS[org], download=download, checksum=None)
        util.unzip(zip_path=zip_path, dst=path)

    return path


def get_morphonet_paths(
    path: Union[os.PathLike, str],
    organism: Optional[Union[str, List[str]]] = None,
    download: bool = False,
) -> List[str]:
    """Get paths to the MorphoNet data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        organism: The organism(s) to use. If None, all organisms will be used.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the h5 data.
    """
    from natsort import natsorted

    if organism is None:
        organism = ORGANISMS
    elif isinstance(organism, str):
        organism = [organism]

    get_morphonet_data(path, organism, download)

    all_h5_paths = []
    for org in organism:
        h5_dir = os.path.join(path, "h5_data", org)
        if not os.path.exists(h5_dir) or len(glob(os.path.join(h5_dir, "*.h5"))) == 0:
            _create_h5_data(path, org)

        h5_paths = glob(os.path.join(h5_dir, "*.h5"))
        all_h5_paths.extend(h5_paths)

    assert len(all_h5_paths) > 0, f"No data found for organism(s) '{organism}'"

    return natsorted(all_h5_paths)


def get_morphonet_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    organism: Optional[Union[str, List[str]]] = None,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the MorphoNet dataset for 3D cell/nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        organism: The organism(s) to use. If None, all organisms will be used.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    h5_paths = get_morphonet_paths(path, organism, download)

    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=True,
    )
    kwargs = util.ensure_transforms(ndim=3, **kwargs)

    return torch_em.default_segmentation_dataset(
        raw_paths=h5_paths,
        raw_key="raw",
        label_paths=h5_paths,
        label_key="labels",
        patch_shape=patch_shape,
        ndim=3,
        **kwargs
    )


def get_morphonet_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int, int],
    organism: Optional[Union[str, List[str]]] = None,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the MorphoNet dataloader for 3D cell/nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        organism: The organism(s) to use. If None, all organisms will be used.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_morphonet_dataset(
        path=path,
        patch_shape=patch_shape,
        organism=organism,
        download=download,
        **ds_kwargs,
    )
    return torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
