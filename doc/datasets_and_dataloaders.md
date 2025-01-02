# Datasets in `torch-em`

We provide PyTorch Datasets / DataLoaders for many different biomedical datasets, mostly for segmentation tasks.
They are implemented in `torch_em.data.datasets`. See `scripts/datasets` for examples on how to visualize images from these datasets.


## Available Datasets

All datasets in `torch_em.data.datasets` are implemented according to the following logic:
- The function `get_..._data` downloads the respective datasets. Note that some datasets cannot be downloaded automatically. In these cases the function will raise an error with a message that explains how to download the data.
- The function `get_..._paths` returns the filepaths to the downloaded inputs.
- The function `get_..._dataset` returns the PyTorch Dataset for the corresponding dataset.
- The function `get_..._dataloader` returns the PyTorch DataLoader for the corresponding dataset.

### Light Microscopy

We provide several light microscopy datasets. See `torch_em.data.datasets.light_microscopy` for an overview.

### Electron Microscopy

We provide several electron microscopy datasets. See `torch_em.data.datasets.electron_microscopy` for an overview.

### Histopathology 

We provide several histopathology datasets. See `torch_em.data.datasets.histopathology` for an overview.

### Medical Imaging

We provide several medical imaging datasets. See `torch_em.data.datasets.medical` for an overview.


## How to create your own dataloader?

Let's say you have a specific dataset of interest and would want to create a PyTorch supported `torch-em`-based dataloader for yourself. We will walk you through how this can be done. See `torch_em/notebooks/tutorial_data_loaders.ipynb` for an extensive tutorial with some examples.

### Supported Data Formats

`torch-em` and [`elf`](https://github.com/constantinpape/elf) currently support Zarr (`.zarr`), NIFTI (`.nii`, `.nii.gz`), HDF5 (`.h5`, `.hdf5`),  N5 (`.n5`), MRC (`.mrc`) and all imageio supported [formats](https://imageio.readthedocs.io/en/v2.5.0/formats.html) (eg. `.tif`, `.png`, `.jpg`, etc.).


### Supported Data Structures

> The recommended input shapes are hinted in all the below mentioned cases as an example.

- 2d images
    - Mono-channel inputs of:
        - ✅ same size (i.e. all images have shape (256, 256), for example)
            - use `SegmentationDataset` (recommended) or `ImageCollectionDataset`
        - ✅ different sizes (i.e. images have shapes like (256, 256), (378, 378), (512, 512), etc., for example)
            - use `ImageCollectionDataset`
    - Multi-channel inputs of:
        - > The ideal expectation of inputs with channels is to have channels first (eg. RGB format -> (256, 256, 3) to channels-first format -> (3, 256, 256))
        - CASE 1: I would like to keep the inputs as RGB format (you must stick to `ImageCollectionDataset` or `is_seg_dataset=False`)
        - CASE 2: I would like to convert the inputs to channels-first (you can be flexible and follow the instructions below)
        - ✅ same size (i.e. all images have shape)
            - use `SegmentationDataset` (recommended for inputs with channels first) or `ImageCollectionDataset` (for inputs in RGB format)
        - ✅ different sizes (i.e. images have shapes like (3, 256, 256), (3, 378, 378), (3, 512, 512), etc., for example)
            - use `ImageCollectionDataset`

- 3d images
    - Mono-channel inputs of:
        - ✅ same size (i.e. all volumes have shape (100, 256, 256), for example)
            - use `SegmentationDataset`
        - ✅ same shape per slice with different z-stack size (i.e. volumes have shape like (100, 256, 256), (100, 256, 256), (100, 256, 256), etc., for example)
            - use `SegmentationDataset` per volume
        - ✅ different sizes (i.e. volumes have shapes like (100, 256, 256), (200, 378, 378), (300, 512, 512), etc., for example)
            -  use `SegmentationDataset` per volume
    - Multi-channel inputs of:
        - ✅ same size (i.e. all volumes have shape (100, 3, 256, 256), for example)
            - use `SegmentationDataset`
        - ✅ same shape per slice with different z-stack size (i.e. volumes have shape like (100, 3, 256, 256), (100, 3, 256, 256), (100, 3, 256, 256), etc., for example)
            - use `SegmentationDataset` per volume
        - ✅ different sizes (i.e. volumes have shapes like (100, 3, 256, 256), (200, 2, 378, 378), (300, 4, 512, 512), etc., for example)
            - use `SegmentationDataset` per volume

#### NOTE:
1. If your data isn't according to one of the suggested data formats, the data loader creation wouldn't work. It's recommended to convert the data into one of the currently supported data structures (we recommend using Zarr / HDF5 / N5 for this purpose) and then move ahead.
2. If your data isn't according to one of the supported data structures, you might run into many issues, leading to incorrect formatting of inputs in your dataloader (we recommend taking a look above at `Supported Data Structures` -> `examples` per point).
3. If you have suggestions (or requests) on additional data formats or data structures, let us know [here](https://github.com/constantinpape/torch-em/issues).

### Create the dataset object

Once you have decided on your choice of dataset class object from above, here's an example on important parameters expected for your custom dataset.

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

### Create the dataloader object

Now that we have our dataset object created, let's finally create the dataloader object to start with the training.

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

### Recommendations

1. Most of the open-source datasets come with their recommended train / val / test splits. In that case, the best practice is to create a function to automatically create the dataset / dataloader for all three splits for you (see `torch_em.data.datasets.dynamicnuclearnet.py` for inspiration) (OR, create three datasets / dataloader one after the other).
2. Some datasets offer a training set and a test set. The best practice is create a "balanced" split internally (for train and val, if desired) and then create the datasets / dataloaders.
3. Some datasets offer only one set of inputs for developing models. There are multiple ways to handle this case, either extend in the direction of `2.`, or design your own heuristic for your use-case.
4. Some datasets offer only training images (without any form of labels). In this case, you could use `RawImageCollectionDataset` as the following (for inspiration, take `torch_em.data.datasets.neurips_cell_seg.py` -> `get_neurips_cellseg_unsupervised_dataset` as reference)

```python
from torch_em.data import RawImageCollectionDataset

dataset = RawImageCollectionDataset(
    raw_image_paths=<LIST_TO_IMAGE_PATHS>,  # path to all images
    # there are other optional parameters, see `torch_em.data.raw_image_collection_dataset.py` for details.
)
```
