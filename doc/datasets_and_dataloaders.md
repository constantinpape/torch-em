# Datasets in `torch-em`

Available open-source datasets in `torch-em` located at `torch_em/data/datasets/` (see `scripts/datasets` for a quick guide on how to use the dataloaders out-of-the-box):

### Microscopy

- ASEM (`asem.py`): Segmentation of organelles in FIB-SEM cells.
- AxonDeepSeg (`axondeepseg.py`): Segmentation of myelinated axons in electron microscopy.
- MitoLab* (`cem.py`):
    - CEM MitoLab: Segmentation of mitochondria in electron microscopy.
    - CEM Mito Benchmark: Segmentation of mitochondria in 7 benchmark electron microscopy datasets.
- Covid IF (`covidif.py`): Segmentation of cells and nuclei in immunofluoroscence.
- CREMI (`cremi.py`): Segmentation of neurons in electron microscopy.
- Cell Tracking Challenge (`ctc.py`): Segmentation data for cell tracking challenge (consists of 10 datasets).
- DeepBacs (`deepbacs.py`): Segmentation of bacteria in light microscopy.
- DSB (`dsb.py`): Segmentation of nuclei in light microscopy.
- DynamicNuclearNet* (`dynamicnuclearnet.py`): Segmentation of nuclei in fluorescence microscopy.
- HPA (`hpa.py`): Segmentation of cells in light microscopy.
- ISBI (`isbi2012.py`): Segmentation of neurons in electron microscopy.
- Kasthuri (`kasthuri.py`): Segmentation of mitochondria in electron microscopy.
- LIVECell (`livecell.py`): Segmentation of cells in phase-contrast microscopy.
- Lucchi (`lucchi.py`): Segmentation of mitochondria in electron microscopy.
- MitoEM (`mitoem.py`): Segmentation of mitochondria in electron microscopy.
- Mouse Embryo (`mouse_embryo.py`): Segmentation of nuclei in confocal microscopy.
- NeurIPS CellSeg (`neurips_cell_seg.py`): Segmentation of cells in multi-modality light microscopy datasets.
- NucMM (`nuc_mm.py`): Segmentation of nuclei in electron microscopy and micro-CT.
- PlantSeg (`plantseg.py`): Segmentation of cells in confocal and light-sheet microscopy.
- Platynereis (`platynereis.py`): Segmentation of nuclei in electron microscopy.
- PNAS* (`pnas_arabidopsis.py`): TODO
- SNEMI (`snemi.py`): Segmentation of neurons in electron microscopy.
- Sponge EM (`sponge_em.py`): Segmentation of sponge cells and organelles in electron microscopy.
- TissueNet* (`tissuenet.py`): Segmentation of cellls in tissue imaged with light microscopy.
- UroCell (`uro_cell.py`): Segmentation of mitochondria and other organelles in electron microscopy.
- VNC (`vnc.py`): Segmentation of mitochondria in electron microscopy

### Histopathology

- BCSS (`bcss.py`): Segmentation of breast cancer tissue in histopathology.
- Lizard* (`lizard.py`): Segmentation of nuclei in histopathology.
- MoNuSaC (`monusac.py`): Segmentation of multi-organ nuclei in histopathology.
- MoNuSeg (`monuseg.py`): Segmentation of multi-organ nuclei in histopathology.
- PanNuke (`pannuke.py`): Segmentation of nuclei in histopathology.


### Medical Imaging

- AutoPET* (`medical/autopet.py`): Segmentation of lesions in whole-body FDG-PET/CT.
- BTCV* (`medical/btcv.py`): Segmentation of multiple organs in CT.

### NOTE:
- \* - These datasets cannot be used out of the box (mostly because of missing automatic downloading). Please take a look at the scripts and the dataset object for details.

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
        - > NOTE: It's important to convert the images to be channels first (see above for the expected format)
        - ✅ same size (i.e. all images have shape (3, 256, 256), for example)
            - use `SegmentationDataset` (recommended) or `ImageCollectionDataset`
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
