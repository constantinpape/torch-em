"""The CATCH dataset contains annotations for tissue segmentation in
H&E stained histopathology images of seven canine cutaneous tumor types.

The dataset consists of 350 whole-slide images (50 per tumor type) with 12,424
polygon annotations across 13 tissue classes. The original Aperio SVS images are
distributed via IBM Aspera (often firewalled), so this loader instead obtains the
images from the Imaging Data Commons (IDC) over HTTPS as DICOM whole-slide images.

This dataset is from the publication https://doi.org/10.1038/s41597-022-01692-w.
Please cite it if you use this dataset in your research. It is hosted on TCIA at
https://doi.org/10.7937/TCIA.2M93-FX66 (CC BY 4.0) and mirrored on IDC.

NOTE: Downloading requires 'idc-index'. Reading the DICOM images requires 'wsidicom'
and rasterizing the polygons requires 'scikit-image'. The data is large (each slide
is around 0.2-2 GB as DICOM), so the slides are downloaded and converted one tumor
type / slide at a time, and the DICOM source is removed after conversion. By default
the full-resolution (base) level is used; this level can be several gigapixels per
slide, so it is read and written to the HDF5 file in tiles. Pass a higher `level` to
use a downsampled level instead.

The annotations are coarse region-level polygons (around 35 per slide), not cell or
nucleus annotations. They are sparse: regions outside any polygon are left as 0, so
'labels/semantic' is a sparse region-level map and class 0 should typically be treated
as background / ignored during training. Each whole-slide image shows a single tumor
type, so within one slide you see that tumor class plus the surrounding normal tissue
classes.

The 13 classes ('labels/semantic') are grouped into a 'Tissue' supercategory (1-6) and
a 'Tumor' supercategory (7-13):
    0: background (unannotated)
    1: Bone
    2: Cartilage
    3: Dermis
    4: Epidermis
    5: Subcutis
    6: Inflamm/Necrosis
    7: Melanoma
    8: Plasmacytoma
    9: Mast Cell Tumor
    10: PNST
    11: SCC
    12: Trichoblastoma
    13: Histiocytoma
"""

import os
import shutil
import zipfile
from glob import glob
from typing import List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

import torch

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


COCO_URL = "https://www.cancerimagingarchive.net/wp-content/uploads/CATCH-json.zip"
IDC_COLLECTION = "catch"
TUMOR_TYPES = ["Histiocytoma", "MCT", "Melanoma", "PNST", "Plasmacytoma", "SCC", "Trichoblastoma"]


def _load_coco(path):
    import json

    coco_path = os.path.join(path, "CATCH.json")
    coco = json.load(open(coco_path))
    annotations = {}
    for ann in coco["annotations"]:
        annotations.setdefault(ann["image_id"], []).append(ann)
    images = {im["file_name"]: (im["id"], im["width"], im["height"]) for im in coco["images"]}
    return images, annotations


def _rasterize_into(label_dataset, annotations, downsample):
    from skimage.draw import polygon as draw_polygon

    height, width = label_dataset.shape
    # Larger regions are drawn first so that smaller annotations stay on top. Each polygon is
    # rasterized within its own bounding box to avoid allocating a full-resolution label in memory.
    for ann in sorted(annotations, key=lambda a: a.get("area", 0), reverse=True):
        segments = ann["segmentation"]
        if segments and isinstance(segments[0], (int, float)):
            segments = [segments]
        for segment in segments:
            xs = np.asarray(segment[0::2], dtype="float64") / downsample
            ys = np.asarray(segment[1::2], dtype="float64") / downsample
            x0, x1 = max(int(np.floor(xs.min())), 0), min(int(np.ceil(xs.max())) + 1, width)
            y0, y1 = max(int(np.floor(ys.min())), 0), min(int(np.ceil(ys.max())) + 1, height)
            if x1 <= x0 or y1 <= y0:
                continue
            rr, cc = draw_polygon(ys - y0, xs - x0, shape=(y1 - y0, x1 - x0))
            block = label_dataset[y0:y1, x0:x1]
            block[rr, cc] = ann["category_id"]
            label_dataset[y0:y1, x0:x1] = block


def _convert_slide(series_uid, file_name, images, annotations, level, output_path, tmp_dir, tile=4096):
    import h5py
    from idc_index import IDCClient
    from wsidicom import WsiDicom

    # Download into a per-series folder, since several slides of the same patient share a PatientID.
    slide_dir = os.path.join(tmp_dir, series_uid)
    if not os.path.exists(slide_dir):
        IDCClient().download_dicom_series(
            seriesInstanceUID=series_uid, downloadDir=tmp_dir, dirTemplate="%SeriesInstanceUID"
        )

    slide = WsiDicom.open(slide_dir)
    try:
        base_width = slide.size.width
        # By default the highest resolution (base) level is used.
        wsi_level = max(slide.levels, key=lambda lv: lv.size.width) if level is None \
            else next(lv for lv in slide.levels if lv.level == level)
        width, height = wsi_level.size.width, wsi_level.size.height
        downsample = base_width / width

        image_id = images[file_name][0]
        tmp_path = output_path + ".tmp"
        with h5py.File(tmp_path, "w") as f:
            raw = f.create_dataset(
                "raw", shape=(3, height, width), dtype="uint8", compression="gzip", chunks=(1, 512, 512)
            )
            label = f.create_dataset(
                "labels/semantic", shape=(height, width), dtype="uint8", compression="gzip", chunks=(512, 512)
            )
            # The base level can be several gigapixels, so the image is read and written in tiles.
            for y in range(0, height, tile):
                for x in range(0, width, tile):
                    th, tw = min(tile, height - y), min(tile, width - x)
                    region = np.array(slide.read_region((x, y), wsi_level.level, (tw, th)))[..., :3]
                    raw[:, y:y + th, x:x + tw] = region.transpose(2, 0, 1)
            _rasterize_into(label, annotations.get(image_id, []), downsample)
    finally:
        slide.close()

    os.replace(tmp_path, output_path)
    shutil.rmtree(slide_dir, ignore_errors=True)


def get_catch_data(
    path: Union[os.PathLike, str],
    tumor_types: Optional[Union[str, List[str]]] = None,
    level: Optional[int] = None,
    download: bool = False,
) -> str:
    """Download and preprocess the CATCH data.

    Args:
        path: Filepath to a folder where the data will be saved.
        tumor_types: The tumor types to use. By default all seven tumor types are used.
        level: The DICOM pyramid level to read. By default the highest resolution (base) level is used.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the folder where the preprocessed data is stored.
    """
    if tumor_types is None:
        tumor_types = TUMOR_TYPES
    if isinstance(tumor_types, str):
        tumor_types = [tumor_types]
    for tumor_type in tumor_types:
        if tumor_type not in TUMOR_TYPES:
            raise ValueError(f"'{tumor_type}' is not a valid tumor type. Choose from {TUMOR_TYPES}.")

    preprocessed_dir = os.path.join(path, "preprocessed")
    tmp_dir = os.path.join(path, "dicom")
    os.makedirs(preprocessed_dir, exist_ok=True)

    coco_path = os.path.join(path, "CATCH.json")
    if not os.path.exists(coco_path):
        zip_path = os.path.join(path, "CATCH-json.zip")
        util.download_source(path=zip_path, url=COCO_URL, download=download, checksum=None)
        with zipfile.ZipFile(zip_path, "r") as f:
            f.extractall(path)

    images, annotations = _load_coco(path)

    try:
        from idc_index import IDCClient
    except ImportError:
        raise ImportError("'idc-index' is required to download CATCH. Install it via conda/pip.")

    # The slide microscopy index provides the 'ContainerIdentifier', which matches the COCO file name
    # ('<ContainerIdentifier>.svs') and is unique per slide (unlike PatientID, where one patient may have
    # several slides).
    client = IDCClient()
    client.fetch_index("sm_index")
    catch = client.index[client.index["collection_id"] == IDC_COLLECTION]
    catch = catch.merge(client.sm_index[["SeriesInstanceUID", "ContainerIdentifier"]], on="SeriesInstanceUID")

    to_convert = catch[catch["ContainerIdentifier"].str.startswith(tuple(tumor_types))]
    for _, row in tqdm(list(to_convert.iterrows()), desc="Converting CATCH slides"):
        container_id = row["ContainerIdentifier"]
        output_path = os.path.join(preprocessed_dir, f"{container_id}.h5")
        if os.path.exists(output_path):
            continue
        if not download:
            raise RuntimeError(f"Cannot find the data at {path}, but download was set to False.")
        _convert_slide(
            row["SeriesInstanceUID"], f"{container_id}.svs", images, annotations, level, output_path, tmp_dir
        )

    return preprocessed_dir


def get_catch_paths(
    path: Union[os.PathLike, str],
    tumor_types: Optional[Union[str, List[str]]] = None,
    level: Optional[int] = None,
    download: bool = False,
) -> List[str]:
    """Get paths to the CATCH data.

    Args:
        path: Filepath to a folder where the data will be saved.
        tumor_types: The tumor types to use. By default all seven tumor types are used.
        level: The DICOM pyramid level to read. By default the highest resolution (base) level is used.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths to the preprocessed HDF5 files.
    """
    preprocessed_dir = get_catch_data(path, tumor_types, level, download)
    volume_paths = sorted(glob(os.path.join(preprocessed_dir, "*.h5")))
    if not volume_paths:
        raise RuntimeError("Could not find any preprocessed CATCH slides for the requested settings.")

    return volume_paths


def get_catch_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    tumor_types: Optional[Union[str, List[str]]] = None,
    level: Optional[int] = None,
    download: bool = False,
    label_dtype: torch.dtype = torch.int64,
    resize_inputs: bool = False,
    **kwargs
) -> Dataset:
    """Get the CATCH dataset for tissue segmentation in canine cutaneous tumor histopathology images.

    Args:
        path: Filepath to a folder where the data will be saved.
        patch_shape: The patch shape to use for training.
        tumor_types: The tumor types to use. By default all seven tumor types are used.
        level: The DICOM pyramid level to read. By default the highest resolution (base) level is used.
        download: Whether to download the data if it is not present.
        label_dtype: The datatype of the labels.
        resize_inputs: Whether to resize the input images.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_catch_paths(path, tumor_types, level, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=volume_paths,
        raw_key="raw",
        label_paths=volume_paths,
        label_key="labels/semantic",
        patch_shape=patch_shape,
        label_dtype=label_dtype,
        is_seg_dataset=True,
        with_channels=True,
        ndim=2,
        **kwargs
    )


def get_catch_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    batch_size: int,
    tumor_types: Optional[Union[str, List[str]]] = None,
    level: Optional[int] = None,
    download: bool = False,
    label_dtype: torch.dtype = torch.int64,
    resize_inputs: bool = False,
    **kwargs
) -> DataLoader:
    """Get the CATCH dataloader for tissue segmentation in canine cutaneous tumor histopathology images.

    Args:
        path: Filepath to a folder where the data will be saved.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        tumor_types: The tumor types to use. By default all seven tumor types are used.
        level: The DICOM pyramid level to read. By default the highest resolution (base) level is used.
        download: Whether to download the data if it is not present.
        label_dtype: The datatype of the labels.
        resize_inputs: Whether to resize the input images.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_catch_dataset(
        path=path, patch_shape=patch_shape, tumor_types=tumor_types, level=level, download=download,
        label_dtype=label_dtype, resize_inputs=resize_inputs, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
