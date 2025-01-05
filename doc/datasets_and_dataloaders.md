# Biomedical Datasets

We provide PyTorch Datasets / DataLoaders for many different biomedical datasets, mostly for segmentation tasks.
They are implemented in `torch_em.data.datasets`. See also [scripts/datasets](https://github.com/constantinpape/torch-em/tree/main/scripts) for examples on how to visualize images from these datasets.

## Available Datasets

All datasets in `torch_em.data.datasets` are implemented according to the following logic:
- The function `get_..._data` downloads the respective datasets. Note that some datasets cannot be downloaded automatically. In these cases the function will raise an error with a message that explains how to manually download the data.
- The function `get_..._paths` returns the filepaths to the downloaded inputs.
- The function `get_..._dataset` returns the PyTorch Dataset for the corresponding dataset.
- The function `get_..._dataloader` returns the PyTorch DataLoader for the corresponding dataset.

We provide ready-to-use light microscopy datasets in `torch_em.data.datasets.light_microscopy`, electron microscopy datasets in `torch_em.data.datasets.electron_microscopy`, histopathology datases in `torch_em.data.datasets.histopathology` and medical imaging datasets in `torch_em.data.datasets.medical`.

## Creating your own Dataset and DataLoader

The following tutorial walks you through the steps to create a `torch_em`-based dataloader for your data.
You can also find an interactive tutorial with examles in [torch_em/notebooks/tutorial_data_loaders.ipynb](https://github.com/constantinpape/torch-em/blob/main/notebooks/tutorial_create_dataloaders.ipynb).

To follow this tutorial you should first familiarize yourself with Datasets and DataLoaders in PyTorch, for example
with the [official PyTorch Tutorial](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html).

### Creating a Dataset

`torch_em` offers two dataset classes for segmentation training: `torch_em.data.ImageCollectionDataset` and `torch_em.data.SegmentationDataset`. Both datasets require image data and segmentation data (to be used as targets for training).
The ImageCollectionDataset supports images of different sizes, but only supports regular image formats such as tif, png or jpeg, the SegmentationDataset supports images of the same size and also supports more complex data formats like hdf5 or zarr.
For an overview of the different input data supported by the two datasets see [Supported Data Formats](supported-data-formats) and [Supported Data Structures](#supported-data-structures).

The simplest way to create one of these datasets is to use the convenience function `torch_em.default_segmentation_dataset`.
You can use the argument `is_segmentation_dataset` to determine whether to use the SegmentationDataset (`True`) or the ImageCollectionDataset (`False`). If this argument is not given, the function will attempt to derive the correct Dataset type from the inputs.

Alternatively, you can also directly instantiate one of the datasets:
```python
from torch_em.data import ImageCollectionDataset, SegmentationDataset

# 1. choice: ImageCollectionDataset
dataset = ImageCollectionDataset(
    raw_image_paths=<SORTED_LIST_OF_IMAGE_PATHS>,  # path to all images
    label_image_paths=<SORTED_LIST_OF_LABEL_PATHS>,  # path to all labels
    patch_shape=<PATCH_SHAPE>,  # the expected patch shape to be extracted from the image
    # there are other optional parameters, see `torch_em.data.image_collection_dataset.py` for details.
)

# 2. choice: SegmentationDataset
dataset = SegmentationDataset(
    raw_path=<PATH_TO_IMAGE>,  # path to one image volume or multiple image volumes (of same shape)
    raw_key=<IMAGE_KEY>,  # the value to access images from heterogenous storage formats like zarr, hdf5, n5
    label_path=<PATH_TO_LABEL>,  # path to one label volume or multiple label volumes (of same shape)
    label_key=<LABEL_KEY>,   # the value to access labels from heterogenous storage formats like zarr, hdf5, n5
    patch_shape=<PATCH_SHAPE>,  # the expected patch shape to be extracted from the image
    ndim=<NDIM>,  # the expected dimension of your desired patches (2 for two-dimensional and 3 for three-dimensional)
    # there are other optional parameters, see `torch_em.data.segmentation_dataset.py` for details.
)
```

### Creating a DataLoader

You can use the convenience function `torch_em.default_segmentation_loader` to directly create a DataLoader.
It will call `torch_em.default_segmentation_dataset` internally.

Alternatively, you can also create a DataLoader from a Dataset object, for example one you have created following the steps outlined in the previous section:

```python
from torch_em.segmentation import get_data_loader

dataset = ...

loader = get_data_loader(
    dataset=dataset,
    batch_size=batch_size,
    # there are other optional parameters, which work the same as for `torch.utils.data.DataLoader`.
    # feel free to pass them with the PyTorch convention, they should work fine.
    # e.g. `shuffle=True`, `num_workers=16`, etc.
)
```

You can now use the DataLoader for training your model, either with `torch_em.default_segmentation_trainer` or with any other PyTorch-based training logic.

### Supported Data Formats

`torch_em` uses [elf](https://github.com/constantinpape/elf) and [imageio](https://imageio.readthedocs.io) to read image data formats.
It thus can open files in the formats supported by `elf`: Zarr (`.zarr`), NIFTI (`.nii`, `.nii.gz`), HDF5 (`.h5`, `.hdf5`),  N5 (`.n5`) and MRC (`.mrc`), and the [formats supported by imageio](https://imageio.readthedocs.io/en/v2.5.0/formats.html) (`.tif`, `.png`, `.jpg`, etc.).

`torch_em.data.SegmentationDataset` supports all of these formats, whereas `torch_em.data.ImageCollectionDataset` only support sthe regular image formats that can be opened with `imageio`.

### Supported Data Structures

> The shapes given in the following are illustative examples.

`torch_em` Datasets and DataLoaders can be created for:
- 2d images:
    - Single-channel inputs of:
        - the same size (all images have the same shape, e.g. (256, 256))
            - use `torch_em.data.SegmentationDataset` (recommended) or `torch_em.data.ImageCollectionDataset`
        - different sizes (images have different shapes, e.g. (256, 256), (378, 378), (512, 512), etc.)
            - use `torch_em.data.ImageCollectionDataset`
    - Multi-channel inputs of:
        - the same same size (i.e. all images have the same shape)
            - use `torch_em.data.SegmentationDataset` (recommended for inputs with channels first, see below) or `torch_em.data.ImageCollectionDataset` (for inputs in RGB format)
        - different sizes (i.e. images have different shapes, e.g. (3, 256, 256), (3, 378, 378), (3, 512, 512), etc.)
            - use `torch_em.data.ImageCollectionDataset`
        - Note: multi-channel inputs are supported best if they have the channel dimension as first axis, (e.g. RGB format -> (256, 256, 3) to channels-first format -> (3, 256, 256)). In order to handle inputs with channel-last / RGB format you can:
            - CASE 1: Keep the inputs in RGB format and use `torch_em.data.ImageCollectionDataset` (or `is_seg_dataset=False`).
            - CASE 2: Convert the inputs to channels-first, see the instructions below.

- 3d images
    - Single-channel inputs of:
        - the same size (all volumes have the same shape, e.g. (100, 256, 256))
            - use `torch_em.data.SegmentationDataset`
        - the same size per slice with a different number of slices (volumes have shapes like (100, 256, 256), (100, 256, 256), (100, 256, 256), etc.)
            - use an individual `torch_em.data.SegmentationDataset` per volume.
        - different sizes (volumes have shapes like (100, 256, 256), (200, 378, 378), (300, 512, 512), etc.)
            - use an individual `torch_em.data.SegmentationDataset` per volume.
    - Multi-channel inputs of:
        - the same size (all volumes have the same shape, e.g. (100, 3, 256, 256))
            - use `torch_em.data.SegmentationDataset`
        - the same size per slice with a different number of slices (volumes have shapes like (100, 3, 256, 256), (100, 3, 256, 256), (100, 3, 256, 256), etc.)
            - use an individual `torch_em.data.SegmentationDataset` per volume.
        - different sizes (volumes have shapes like (100, 3, 256, 256), (200, 2, 378, 378), (300, 4, 512, 512), etc.)
            - use an individual `torch_em.data.SegmentationDataset` per volume.

You can create a combined dataset out of multiple individual datasets using `torch_em.data.ConcatDataset`.
You can also use `torch_em.default_segmentation_dataset` / `torch_em.default_segmentation_loader` and pass
a list of file paths to the `raw_paths` and `label_paths` arguments.
This will create multiple datasets internally and then combine them.

Note:
1. If your data isn't according to one of the suggested data formats, the DataLoader creation probably won't work. It's recommended to convert the data into one of the currently supported data structures (we recommend using Zarr / HDF5 / N5 for this purpose) and then move ahead.
2. If your data isn't according to one of the supported data structures, the data loader might stil work, but you will run into issues leater, leading to incorrect formatting of inputs in your dataloader.
3. If you have suggestions (or requests) for additional data formats or data structures, let us know [here](https://github.com/constantinpape/torch-em/issues).

### Further Recommendations

1. Most of the open-source datasets come with their recommended train / val / test splits. In that case, the best practice is to create a function to automatically create the dataset / dataloader for all three splits for you (see `torch_em.data.datasets.dynamicnuclearnet` for an example) (OR, create three datasets / dataloader one after the other).
2. Some datasets offer a training set and a test set. The best practice is create a "balanced" split internally (for train and val, if desired) and then create the datasets / dataloaders.
3. Some datasets offer only one set of inputs for developing models. There are multiple ways to handle this case, either extend in the direction of `2.`, or design your own heuristic for your use-case.
4. Some datasets offer only training images (without any form of labels). In this case, you could use `torch_em.data.RawDataset` or `torch_em.data.RawImageCollectionDataset`, see for example `torch_em.data.datasets.neurips_cell_seg.get_neurips_cellseg_unsupervised_dataset`. Below is a small snippet showing how to use the RawImageCollectionDataset.

```python
from torch_em.data import RawImageCollectionDataset

dataset = RawImageCollectionDataset(
    raw_image_paths=<LIST_TO_IMAGE_PATHS>,  # path to all images
    # there are other optional parameters, see `torch_em.data.raw_image_collection_dataset.py` for details.
)
```
