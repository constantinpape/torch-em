import os
from glob import glob
from tqdm import tqdm
from natsort import natsorted

import numpy as np

import torch_em

from .. import util


def get_duke_liver_data(path, download):
    """The dataset is located at https://doi.org/10.5281/zenodo.7774566.

    Follow the instructions below to get access to the dataset.
    - Visit the zenodo site attached above.
    - Send a request message alongwith some details to get access to the dataset.
    - The authors would accept the request, then you can access the dataset.
    - Next, download the `Segmentation.zip` file and provide the path where the zip file is stored.
    """
    if download:
        raise NotImplementedError(
            "Automatic download for Duke Liver dataset is not possible. See `get_duke_liver_data` for details."
        )

    data_dir = os.path.join(path, "data", "Segmentation")
    if os.path.exists(data_dir):
        return data_dir

    zip_path = os.path.join(path, "Segmentation.zip")
    util.unzip(zip_path=zip_path, dst=os.path.join(path, "data"), remove=False)
    return data_dir


def _preprocess_data(path, data_dir):
    preprocess_dir = os.path.join(path, "data", "preprocessed")

    if os.path.exists(preprocess_dir):
        _image_paths = natsorted(glob(os.path.join(preprocess_dir, "images", "*.nii.gz")))
        _gt_paths = natsorted(glob(os.path.join(preprocess_dir, "masks", "*.nii.gz")))
        return _image_paths, _gt_paths

    os.makedirs(os.path.join(preprocess_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(preprocess_dir, "masks"), exist_ok=True)

    import pydicom as dicom
    import nibabel as nib

    image_paths, gt_paths = [], []
    for patient_dir in tqdm(glob(os.path.join(data_dir, "00*"))):
        patient_id = os.path.split(patient_dir)[-1]

        for sub_id_dir in glob(os.path.join(patient_dir, "*")):
            sub_id = os.path.split(sub_id_dir)[-1]

            image_path = os.path.join(preprocess_dir, "images", f"{patient_id}_{sub_id}.nii.gz")
            gt_path = os.path.join(preprocess_dir, "masks", f"{patient_id}_{sub_id}.nii.gz")

            image_paths.append(image_path)
            gt_paths.append(gt_path)

            if os.path.exists(image_path) and os.path.exists(gt_path):
                continue

            image_slice_paths = natsorted(glob(os.path.join(sub_id_dir, "images", "*.dicom")))
            gt_slice_paths = natsorted(glob(os.path.join(sub_id_dir, "masks", "*.dicom")))

            images, gts = [], []
            for image_slice_path, gt_slice_path in zip(image_slice_paths, gt_slice_paths):
                image_slice = dicom.dcmread(image_slice_path).pixel_array
                gt_slice = dicom.dcmread(gt_slice_path).pixel_array

                images.append(image_slice)
                gts.append(gt_slice)

            image = np.stack(images).transpose(1, 2, 0)
            gt = np.stack(gts).transpose(1, 2, 0)

            assert image.shape == gt.shape

            image = nib.Nifti2Image(image, np.eye(4))
            gt = nib.Nifti2Image(gt, np.eye(4))

            nib.save(image, image_path)
            nib.save(gt, gt_path)

    return image_paths, gt_paths


def _get_duke_liver_paths(path, split, download):
    data_dir = get_duke_liver_data(path=path, download=download)

    image_paths, gt_paths = _preprocess_data(path=path, data_dir=data_dir)

    if split == "train":
        image_paths, gt_paths = image_paths[:250], gt_paths[:250]
    elif split == "val":
        image_paths, gt_paths = image_paths[250:260], gt_paths[250:260]
    elif split == "test":
        image_paths, gt_paths = image_paths[260:], gt_paths[260:]
    else:
        raise ValueError(f"'{split}' is not a valid split.")

    return image_paths, gt_paths


def get_duke_liver_dataset(
    path,
    patch_shape,
    split,
    resize_inputs=False,
    download=False,
    **kwargs
):
    """Dataset for segmentation of liver in MRI.

    This dataset is from Macdonald et al. - https://doi.org/10.1148/ryai.220275.

    The dataset is located at https://doi.org/10.5281/zenodo.7774566 (see `get_duke_liver_dataset` for details).

    Please cite it if you use it in a publication.
    """

    image_paths, gt_paths = _get_duke_liver_paths(path=path, split=split, download=download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    dataset = torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key="data",
        label_paths=gt_paths,
        label_key="data",
        is_seg_dataset=True,
        patch_shape=patch_shape,
        **kwargs
    )

    return dataset


def get_duke_liver_loader(
    path,
    patch_shape,
    batch_size,
    split,
    resize_inputs=False,
    download=False,
    **kwargs
):
    """Dataloader for segmentation of liver in MRI. See `get_duke_liver_dataset` for details.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_duke_liver_dataset(
        path=path, patch_shape=patch_shape, split=split, resize_inputs=resize_inputs, download=download, **ds_kwargs
    )
    loader = torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
    return loader
