"""The OrgLine dataset contains organoid images and associated segmentation masks.

The organoids come from different organs and were assembled from different prior publications.
Specifically:
- Intestine: from OrgaQuant (https://doi.org/10.1038/s41598-019-48874-y)
             from OrgaSegment (https://doi.org/10.1038/s42003-024-05966-4)
- Brain:     from  https://doi.org/10.1038/s41597-024-03330-z
- Colon:     from OrgaExtractor (https://doi.org/10.1038/s41598-023-46485-2)
- PDAC:      from OrganoID (https://doi.org/10.1371/journal.pcbi.1010584)
             from OrganoidNet (https://doi.org/10.1007/s13402-024-00958-2)
- Stomach:   from https://zenodo.org/records/18447547
- Breast:    from https://zenodo.org/records/18447547

Please cite the associated zenodo entry (https://zenodo.org/records/16355179) and the relevant original publications
if you use this dataset for your research.
"""

import os
import shutil
from glob import glob
from typing import Union, Tuple, List, Literal, Optional, Sequence

import h5py
import imageio.v3 as imageio
import numpy as np

from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

import torch_em

from .. import util

try:
    from pycocotools.coco import COCO
except ImportError:
    COCO = None


URL1 = "https://zenodo.org/records/16355179/files/InstanceSeg.zip?download=1"
URL2 = "https://zenodo.org/records/18447547/files/data.zip?download=1"

CHECKSUM1 = "6787dc47ee5f800e7ecf4a51d958fc88591c877ca7f8f03c2aa3e7fa7c4aca50"
CHECKSUM2 = "8b5984ee19232c06cdf5366080a3f3b27fb2109f38a2a345316e22dd2bb9a1c2"

ORGANS1 = ("PDAC", "colon", "Intestine", "brain")
ORGANS2 = ("stomach", "breast")


def _annotations_to_instances(coco, image_metadata):
    from skimage.measure import label
    from skimage.segmentation import relabel_sequential

    # create and save the segmentation
    annotation_ids = coco.getAnnIds(imgIds=image_metadata["id"])
    annotations = coco.loadAnns(annotation_ids)
    assert len(annotations) <= np.iinfo("uint16").max
    shape = (image_metadata["height"], image_metadata["width"])
    seg = np.zeros(shape, dtype="uint32")

    sizes = [ann["area"] for ann in annotations]
    sorting = np.argsort(sizes)
    annotations = [annotations[i] for i in sorting]

    for seg_id, annotation in enumerate(annotations, 1):
        mask = coco.annToMask(annotation).astype("bool")
        assert mask.shape == seg.shape
        seg[mask] = seg_id

    # Filter out small pieces from pasting organoids on top of each other.
    min_size = 25
    seg = label(seg)
    seg_ids, sizes = np.unique(seg, return_counts=True)
    seg[np.isin(seg, seg_ids[sizes < min_size])] = 0
    seg, _, _ = relabel_sequential(seg)

    return seg.astype("uint16")


def _prepare_data(data_dir, organ):
    if organ in ORGANS1:
        for org in ORGANS1:
            input_root, output_root = os.path.join(data_dir, "InstanceSeg", org), os.path.join(data_dir, org)
            for split in ("train", "val", "test"):
                images = sorted(glob(os.path.join(input_root, split, "images", "*")))
                masks = sorted(glob(os.path.join(input_root, split, "masks", "*")))
                if len(images) != len(masks):
                    continue
                assert len(images) == len(masks)
                output_folder = os.path.join(output_root, split)
                os.makedirs(output_folder, exist_ok=True)
                for im_path, mask_path in tqdm(
                    zip(images, masks), total=len(images), desc=f"Converting {org}, {split}-split"
                ):
                    im = imageio.imread(im_path)
                    mask = np.load(mask_path) if mask_path.endswith(".npy") else imageio.imread(mask_path)
                    if im.ndim == 3:
                        im = im[..., 0]
                    assert im.shape == mask.shape
                    out_path = os.path.join(output_folder, f"{os.path.basename(im_path)}.h5")
                    with h5py.File(out_path, mode="w") as f:
                        f.create_dataset("image", data=im, compression="gzip")
                        f.create_dataset("masks", data=mask, compression="gzip")
        shutil.rmtree(os.path.join(data_dir, "InstanceSeg"))

    else:
        if COCO is None:
            raise ModuleNotFoundError(
                "'pycocotools' is required for processing the OrgLine ground-truth. "
                "Install it with 'conda install -c conda-forge pycocotools'."
            )
        for org in ORGANS2:
            input_root, output_root = os.path.join(data_dir, org), os.path.join(data_dir, org)
            coco_file = os.path.join(input_root, "coco.json")
            coco = COCO(coco_file)

            image_ids = coco.getImgIds()
            # Create splits.
            train_ids, test_ids = train_test_split(image_ids, test_size=0.2, random_state=42)
            test_ids, val_ids = train_test_split(test_ids, test_size=0.6, random_state=42)
            train_out, val_out = os.path.join(output_root, "train"), os.path.join(output_root, "val")
            test_out = os.path.join(output_root, "test")
            os.makedirs(train_out, exist_ok=True)
            os.makedirs(val_out, exist_ok=True)
            os.makedirs(test_out, exist_ok=True)

            for image_id in tqdm(image_ids, desc=f"Converting {org}"):
                image_metadata = coco.loadImgs(image_id)[0]
                file_name = image_metadata["file_name"]
                image_path = os.path.join(input_root, file_name)
                im = imageio.imread(image_path)
                if im.ndim == 3:
                    im = np.mean(im[..., :3], axis=-1)
                mask = _annotations_to_instances(coco, image_metadata)
                assert im.shape == mask.shape
                # For debugging.
                # import napari
                # v = napari.Viewer()
                # v.add_image(im)
                # v.add_labels(mask)
                # napari.run()
                if image_id in train_ids:
                    output_folder = train_out
                elif image_id in val_ids:
                    output_folder = val_out
                else:
                    output_folder = test_out
                out_path = os.path.join(output_folder, f"{os.path.basename(image_path)}.h5")
                with h5py.File(out_path, mode="w") as f:
                    f.create_dataset("image", data=im, compression="gzip")
                    f.create_dataset("masks", data=mask, compression="gzip")

            # Clean up.
            shutil.rmtree(os.path.join(input_root, "images"))
            json_files = glob(os.path.join(input_root, "*.json"))
            for json_file in json_files:
                os.remove(json_file)


def get_orgline_data(path: Union[os.PathLike, str], organ: str,  download: bool = False) -> str:
    """Download the OrgLine dataset.

    Args:
        path: Filepath to the folder where the downloaded data will be saved.
        organ: The organ from which the organoids are derived.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath where the data is downloaded.
    """
    if organ in ORGANS1:
        url, checksum = URL1, CHECKSUM1
        data_folder = "data1"
    elif organ in ORGANS2:
        url, checksum = URL2, CHECKSUM2
        data_folder = "data2"
    else:
        raise ValueError(f"Invalid organ: {organ}. Must be one of {ORGANS1 + ORGANS2}.")

    data_dir = os.path.join(path, data_folder)
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "data.zip")
    util.download_source(path=zip_path, url=url, download=download, checksum=checksum)
    util.unzip(zip_path=zip_path, dst=data_dir, remove=True)
    _prepare_data(data_dir, organ)
    return data_dir


def get_orgline_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "val", "test"],
    organs: Optional[Union[str, Sequence[str]]] = None,
    download: bool = False,
) -> List[str]:
    """Get paths to the OrgLine data.

    Args:
        path: Filepath to the folder where the downloaded data will be saved.
        organ: .
        split: The data split to use.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the input data.
    """
    if isinstance(organs, str):
        organs = [organs]
    elif organs is None:
        organs = ORGANS1 + ORGANS2
    paths = []
    for organ in organs:
        data_dir = get_orgline_data(path, organ, download)
        this_paths = sorted(glob(os.path.join(data_dir, organ, split, "*.h5")))
        paths.extend(this_paths)
    return paths


def get_orgline_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"],
    organs: Optional[Union[str, Sequence[str]]] = None,
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get OrgLine dataset for organoid segmentation in brightfield microscopy images.

    Args:
        path: Filepath to the folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        split: The data split to use.
        organ:
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    paths = get_orgline_paths(path, split, organs, download)
    return torch_em.default_segmentation_dataset(
        raw_paths=paths,
        raw_key="image",
        label_paths=paths,
        label_key="masks",
        is_seg_dataset=True,
        patch_shape=patch_shape,
        ndim=2,
        **kwargs
    )


def get_orgline_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"],
    organs: Optional[Union[str, Sequence[str]]] = None,
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get OrgLine dataloader for organoid segmentation in brightfield microscopy images.

    Args:
        path: Filepath to the folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.

        split: The data split to use.

        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_orgline_dataset(path, patch_shape, split=split, organs=organs, download=download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
