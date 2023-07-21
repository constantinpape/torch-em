import json
import os
from glob import glob

import numpy as np
import torch
import torch_em


def to_rgb(image):
    if image.ndim == 2:
        image = np.concatenate([image[None]] * 3, axis=0)
    assert image.ndim == 3
    assert image.shape[0] == 3, f"{image.shape}"
    return image


# would be better to make balanced splits for the different data modalities
# (but we would need to know mapping of images to modality)
def _get_image_and_label_paths(root, split, val_fraction):
    path = os.path.join(root, "TrainLabeled")
    assert os.path.exists(root), root

    image_folder = os.path.join(path, "images")
    assert os.path.exists(image_folder)
    label_folder = os.path.join(path, "labels")
    assert os.path.exists(label_folder)

    all_image_paths = glob(os.path.join(image_folder, "*"))
    all_image_paths.sort()
    all_label_paths = glob(os.path.join(label_folder, "*"))
    all_label_paths.sort()
    assert len(all_image_paths) == len(all_label_paths)

    if split is None:
        return all_image_paths, all_label_paths

    split_file = os.path.join(
        os.path.split(__file__)[0], f"split_{val_fraction}.json"
    )

    if os.path.exists(split_file):
        with open(split_file) as f:
            split_ids = json.load(f)[split]
    else:
        # split into training and val images
        n_images = len(all_image_paths)
        n_train = int((1.0 - val_fraction) * n_images)
        image_ids = list(range(n_images))
        np.random.shuffle(image_ids)
        train_ids, val_ids = image_ids[:n_train], image_ids[n_train:]
        assert len(train_ids) + len(val_ids) == n_images

        with open(split_file, "w") as f:
            json.dump({"train": train_ids, "val": val_ids}, f)

        split_ids = val_ids if split == "val" else train_ids

    image_paths = [all_image_paths[idx] for idx in split_ids]
    label_paths = [all_label_paths[idx] for idx in split_ids]
    assert len(image_paths) == len(label_paths)
    return image_paths, label_paths


def get_neurips_cellseg_supervised_dataset(
    root, split, patch_shape,
    make_rgb=True,
    label_transform=None,
    label_transform2=None,
    raw_transform=None,
    transform=None,
    label_dtype=torch.float32,
    dtype=torch.float32,
    n_samples=None,
    sampler=None,
    val_fraction=0.1,
):
    """Dataset for the segmentation of cells in light microscopy.

    This dataset is part of the NeuRIPS Cell Segmentation challenge: https://neurips22-cellseg.grand-challenge.org/.
    """
    assert split in ("train", "val", None), split
    image_paths, label_paths = _get_image_and_label_paths(root, split, val_fraction)

    if raw_transform is None:
        trafo = to_rgb if make_rgb else None
        raw_transform = torch_em.transform.get_raw_transform(augmentation2=trafo)
    if transform is None:
        transform = torch_em.transform.get_augmentations(ndim=2)

    ds = torch_em.data.ImageCollectionDataset(image_paths, label_paths,
                                              patch_shape=patch_shape,
                                              raw_transform=raw_transform,
                                              label_transform=label_transform,
                                              label_transform2=label_transform2,
                                              label_dtype=label_dtype,
                                              transform=transform,
                                              n_samples=n_samples,
                                              sampler=sampler)
    return ds


def get_neurips_cellseg_supervised_loader(
    root, split,
    patch_shape, batch_size,
    make_rgb=True,
    label_transform=None,
    label_transform2=None,
    raw_transform=None,
    transform=None,
    label_dtype=torch.float32,
    dtype=torch.float32,
    n_samples=None,
    sampler=None,
    val_fraction=0.1,
    **loader_kwargs
):
    """Dataloader for the segmentation of cells in light microscopy. See 'get_neurips_cellseg_supervised_dataset'."""
    ds = get_neurips_cellseg_supervised_dataset(
        root, split, patch_shape, make_rgb=make_rgb, label_transform=label_transform,
        label_transform2=label_transform2, raw_transform=raw_transform, transform=transform,
        label_dtype=label_dtype, dtype=dtype, n_samples=n_samples, sampler=sampler, val_fraction=val_fraction,
    )
    return torch_em.segmentation.get_data_loader(ds, batch_size, **loader_kwargs)


def _get_image_paths(root):
    path = os.path.join(root, "TrainUnlabeled")
    assert os.path.exists(path), path
    image_paths = glob(os.path.join(path, "*"))
    image_paths.sort()
    return image_paths


def _get_wholeslide_paths(root, patch_shape):
    path = os.path.join(root, "TrainUnlabeled_WholeSlide")
    assert os.path.exists(path), path
    image_paths = glob(os.path.join(path, "*"))
    image_paths.sort()

    # one of the whole slides doesn't support memmap which will make it very slow to load
    image_paths = [path for path in image_paths if torch_em.util.supports_memmap(path)]
    assert len(image_paths) > 0

    n_samples = 0
    for im_path in image_paths:
        shape = torch_em.util.load_image(im_path).shape
        assert len(shape) == 3 and shape[-1] == 3
        im_shape = shape[:2]
        n_samples += np.prod([sh // psh for sh, psh in zip(im_shape, patch_shape)])

    return image_paths, n_samples


def get_neurips_cellseg_unsupervised_dataset(
    root, patch_shape,
    make_rgb=True,
    raw_transform=None,
    transform=None,
    dtype=torch.float32,
    sampler=None,
    use_images=True,
    use_wholeslide=True,
):
    """Dataset for the segmentation of cells in light microscopy.

    This dataset is part of the NeuRIPS Cell Segmentation challenge: https://neurips22-cellseg.grand-challenge.org/.
    """
    if raw_transform is None:
        trafo = to_rgb if make_rgb else None
        raw_transform = torch_em.transform.get_raw_transform(augmentation2=trafo)
    if transform is None:
        transform = torch_em.transform.get_augmentations(ndim=2)

    datasets = []
    if use_images:
        image_paths = _get_image_paths(root)
        datasets.append(torch_em.data.RawImageCollectionDataset(image_paths,
                                                                patch_shape=patch_shape,
                                                                raw_transform=raw_transform,
                                                                transform=transform,
                                                                dtype=dtype,
                                                                sampler=sampler))
    if use_wholeslide:
        image_paths, n_samples = _get_wholeslide_paths(root, patch_shape)
        datasets.append(torch_em.data.RawImageCollectionDataset(image_paths,
                                                                patch_shape=patch_shape,
                                                                raw_transform=raw_transform,
                                                                transform=transform,
                                                                dtype=dtype,
                                                                n_samples=n_samples,
                                                                sampler=sampler))
    assert len(datasets) > 0
    return torch.utils.data.ConcatDataset(datasets)


def get_neurips_cellseg_unsupervised_loader(
    root, patch_shape, batch_size,
    make_rgb=True,
    raw_transform=None,
    transform=None,
    dtype=torch.float32,
    sampler=None,
    use_images=True,
    use_wholeslide=True,
    **loader_kwargs,
):
    """Dataloader for the segmentation of cells in light microscopy. See 'get_neurips_cellseg_unsupervised_dataset'."""
    ds = get_neurips_cellseg_unsupervised_dataset(
        root, patch_shape, make_rgb=make_rgb, raw_transform=raw_transform, transform=transform,
        dtype=dtype, sampler=sampler, use_images=use_images, use_wholeslide=use_wholeslide
    )
    return torch_em.segmentation.get_data_loader(ds, batch_size, **loader_kwargs)
