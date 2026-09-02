"""SPATCH contains expert annotations for nucleus segmentation in images of human tumor tissue.

The dataset covers three tumor types (ovarian cancer, hepatocellular carcinoma and colon adenocarcinoma)
imaged on four spatial transcriptomics platforms. Xenium and CosMx provide DAPI images, while Visium HD
and Stereo-seq provide H&E images. Every subset ships five manually annotated tiles.
NOTE: The annotations mark nuclear boundaries. They come as polygons, which this module rasterizes.

The dataset is hosted at https://spatch.pku-genomics.org.
This dataset is from the publication https://doi.org/10.1038/s41467-025-64292-3.
Please cite it if you use this dataset for your research.
"""

import os
import json
import zipfile
import urllib.request
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List

import numpy as np
import imageio.v3 as imageio
from skimage.draw import polygon
from skimage.segmentation import relabel_sequential

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


TOKEN_URL = "https://bj21400.api.aliyunfile.com/v2/share_link/get_share_token"
LIST_URL = "https://bj21400.api.aliyunfile.com/v2/file/list"

# The portal serves every archive through a share link, which is addressed by the id below.
SHARE_IDS = {
    "xenium_ov": "PXSYun7LWov",
    "xenium_hcc": "mzaFwMVEuvq",
    "xenium_coad": "TUzo1AaZ9Vx",
    "cosmx_ov": "ttidbCgTY1z",
    "cosmx_hcc": "aeubu9Uxt2Z",
    "cosmx_coad": "Ja4mTzWciZb",
    "visium_hd_ov": "Joubkps1Mz1",
    "visium_hd_hcc": "sY36AC8GoPW",
    "visium_hd_coad": "PjBvUPtFaqP",
    "stereoseq_ov": "3Es7YQz69ja",
}

CHECKSUMS = {
    "xenium_ov": "674a7e317c75d5718bd42d8fdcc11fd1d0f9f7274b597e074c453a14de83452e",
    "xenium_hcc": "256052bfc589be0c765d69ec884568dad54b1e395e32943a155388a087f93f72",
    "xenium_coad": "a3c04a7f8d307508e8f4382483341e6fb69f3f4c4630ca54f43c6ee64792c4b5",
    "cosmx_ov": "040fd2d29a0fc278c0ef46c231d6825c049c6c87feb31c5a6d96bdaf9ff09d37",
    "cosmx_hcc": "39dccac329d50218524429002c246681310dfb15091eaa8431fafe8d1c318aa3",
    "cosmx_coad": "dda1af15d0e2c191a83d9f518661a420cb51716e428b860612a60d5155e5ec2f",
    "visium_hd_ov": "3e25df5fb9964a95158810bbb621b988fdcaa81de428d9a5f6c34ee0aa6099e7",
    "visium_hd_hcc": "322eda304ec22e49037b35bd3d60c3d565c4dd202609c5c570a05f4672beedb9",
    "visium_hd_coad": "7aa9e5da53fc8aa83199f10383e945aa85f147e4be3d8db61b824d13cbe05441",
    "stereoseq_ov": "4187167c8d622f8f187d5d0f30bb3ce599154e2b01d8e68147faa9042b12e88d",
}


def _post_json(url, payload, headers=None):
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST")
    request.add_header("Content-Type", "application/json")
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read())


def _get_download_url(share_id):
    """Resolve the share id of a subset to a download url, which the portal only issues on request."""
    token = _post_json(TOKEN_URL, {"share_id": share_id, "ignoreError": True})["share_token"]
    payload = {
        "limit": 100,
        "marker": "",
        "share_id": share_id,
        "parent_file_id": "root",
        "fields": "user_name,dir_size,url,content_type,upload_id,crc64_hash,revision_id,description",
        "url_expire_sec": 7200,
    }
    items = _post_json(LIST_URL, payload, {"x-share-token": token})["items"]
    if not items:
        raise RuntimeError(f"The share link '{share_id}' does not contain any file.")
    return items[0]["download_url"]


def _get_outlines(json_path):
    """Return the nucleus outlines, which come in the labelme format, or in the darwin format.

    Some tiles annotate the same nucleus twice, so this drops the repeated outlines.
    """
    with open(json_path) as f:
        content = json.load(f)

    if "annotations" in content:  # darwin
        paths = [path for a in content["annotations"] for path in a.get("polygon", {}).get("paths", [])]
        outlines = [np.array([[p["y"], p["x"]] for p in path]) for path in paths]
    else:
        shapes = [s for s in content["shapes"] if s.get("shape_type") == "polygon"]
        outlines = [np.array([[p[1], p[0]] for p in s["points"]]) for s in shapes]

    seen, unique = set(), []
    for outline in outlines:
        key = np.round(outline, 4).tobytes()
        if key not in seen:
            seen.add(key)
            unique.append(outline)
    return unique


def _get_instances(json_path, shape):
    """Paint the outlines into a label image.

    Nuclei that the annotators drew on top of each other cannot all survive this, so the labels
    are made consecutive afterwards.
    """
    outlines = _get_outlines(json_path)
    instances = np.zeros(shape, dtype="uint16")
    for label, outline in enumerate(outlines, start=1):
        rr, cc = polygon(outline[:, 0], outline[:, 1], shape=shape)
        instances[rr, cc] = label
    return relabel_sequential(instances)[0]


def _preprocess_data(input_dir, data_dir):
    import h5py

    os.makedirs(data_dir, exist_ok=True)
    tile_paths = natsorted(glob(os.path.join(input_dir, "**", "tile*.png"), recursive=True))
    if not tile_paths:
        raise RuntimeError(f"Could not find the annotated tiles in {input_dir}.")

    for tile_path in tile_paths:
        tile = os.path.splitext(os.path.basename(tile_path))[0]
        json_path = os.path.join(os.path.dirname(tile_path), f"{tile.replace('tile', 'mask')}.json")
        out_path = os.path.join(data_dir, f"{tile}.h5")
        if os.path.exists(out_path) or not os.path.exists(json_path):
            continue

        image = imageio.imread(tile_path)[..., :3]
        instances = _get_instances(json_path, image.shape[:2])

        with h5py.File(out_path, "a") as f:
            f.create_dataset("raw/rgb", data=image.transpose(2, 0, 1), compression="gzip")
            f.create_dataset("labels/nuclei", data=instances, compression="gzip")


def get_spatch_data(path: Union[os.PathLike, str], subset: str, download: bool = False) -> str:
    """Download one subset of the SPATCH dataset.

    Args:
        path: The folder where the function stores the data.
        subset: The subset of the dataset. See `SHARE_IDS` for the valid choices.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the folder with the prepared data.
    """
    if subset not in SHARE_IDS:
        raise ValueError(f"'{subset}' is not a valid subset. Choose one of {list(SHARE_IDS.keys())}.")

    data_dir = os.path.join(path, subset, "data")
    if glob(os.path.join(data_dir, "*.h5")):
        return data_dir

    subset_dir = os.path.join(path, subset)
    os.makedirs(subset_dir, exist_ok=True)

    input_dir = os.path.join(subset_dir, "manual_segmentation")
    if not os.path.exists(input_dir):
        zip_path = os.path.join(subset_dir, f"{subset}.zip")
        if not os.path.exists(zip_path):
            if not download:
                raise RuntimeError(f"The data for '{subset}' is not present and download is set to False.")
            util.download_source(
                path=zip_path, url=_get_download_url(SHARE_IDS[subset]), download=True, checksum=CHECKSUMS[subset]
            )
        with zipfile.ZipFile(zip_path) as f:
            f.extractall(subset_dir)

    _preprocess_data(input_dir, data_dir)

    return data_dir


def get_spatch_paths(
    path: Union[os.PathLike, str], subset: Union[str, List[str]], download: bool = False
) -> List[str]:
    """Get the paths to the SPATCH data.

    Args:
        path: The folder where the function stores the data.
        subset: One subset or a list of subsets. See `SHARE_IDS` for the valid choices.
        download: Whether to download the data if it is not present.

    Returns:
        The list of filepaths to the input data.
    """
    subsets = [subset] if isinstance(subset, str) else subset
    volume_paths = []
    for name in subsets:
        data_dir = get_spatch_data(path, name, download)
        volume_paths.extend(natsorted(glob(os.path.join(data_dir, "*.h5"))))

    assert len(volume_paths) > 0, f"Could not find data for the subset '{subset}'."
    return volume_paths


def get_spatch_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    subset: Union[str, List[str]],
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the SPATCH dataset for nucleus segmentation in images of human tumor tissue.

    Args:
        path: The folder where the function stores the data.
        patch_shape: The patch shape to use for training.
        subset: One subset or a list of subsets. See `SHARE_IDS` for the valid choices.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_spatch_paths(path, subset, download)
    kwargs = util.update_kwargs(kwargs, "with_channels", True)

    return torch_em.default_segmentation_dataset(
        raw_paths=volume_paths,
        raw_key="raw/rgb",
        label_paths=volume_paths,
        label_key="labels/nuclei",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        ndim=2,
        **kwargs
    )


def get_spatch_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    subset: Union[str, List[str]],
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the SPATCH dataloader for nucleus segmentation in images of human tumor tissue.

    Args:
        path: The folder where the function stores the data.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        subset: One subset or a list of subsets. See `SHARE_IDS` for the valid choices.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_spatch_dataset(path, patch_shape, subset, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size=batch_size, **loader_kwargs)
