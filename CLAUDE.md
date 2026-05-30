# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`torch_em` is a PyTorch library for deep-learning based semantic and instance segmentation of bioimage data (electron microscopy, light microscopy, histopathology, medical imaging). It provides a functional API to train segmentation models with few lines of code, plus a large collection of ready-to-use datasets.

## Environment & Commands

Dependencies are NOT all pip-installable (e.g. `affogato`, `nifty`, `python-elf`, `bioimageio.core`, `kornia`). Use conda/mamba with `environment.yaml`:

```bash
mamba env create -f environment.yaml      # creates env "torch-em-dev"
pip install --no-deps -e .                # install package in editable mode
```

Run the test suite (CI uses Python 3.10/3.11 and plain unittest, not pytest):

```bash
python -m unittest discover -s test -v          # all tests
python -m unittest test.test_segmentation       # single test module
python -m unittest test.data.test_sampler.TestSampler.test_min_foreground   # single test
```

Build docs locally: `pdoc torch_em/ -d google -o doc/` (CI deploys these to GitHub Pages).

## Architecture & Conventions

**Code-only configuration.** All parameters are specified in Python — there are no config files. To extend training, *inherit from `torch_em.trainer.DefaultTrainer`* rather than adding callbacks (the design deliberately has no callback system). All data loading is lazy to support training on datasets larger than memory.

**Three-layer API.** Most users go through convenience functions in `torch_em/segmentation.py`, re-exported at the top level:
- `default_segmentation_dataset` → builds a `SegmentationDataset` or `ImageCollectionDataset` (auto-detected from inputs via `is_segmentation_dataset`).
- `default_segmentation_loader` → wraps the dataset in a DataLoader.
- `default_segmentation_trainer` → returns a pre-configured `DefaultTrainer`.

`DefaultTrainer.fit(iterations=...)` runs the train/val loop. Checkpoints are saved to `./checkpoints/<name>/` and tensorboard logs to `./logs/<name>/` (override the base dir with `save_root`). The trainer records its own constructor arguments (`get_constructor_arguments`) so a run can be fully reconstructed and resumed from a checkpoint.

**Two dataset classes** (`torch_em/data/`):
- `SegmentationDataset` — same-shape volumes, supports complex formats (hdf5, zarr, n5) addressed via a `*_key`. Used for 2D and 3D.
- `ImageCollectionDataset` — variably-sized images, only regular formats (tif/png/jpeg).

Both pair a raw input with a label target. A raw "path" may be a list of paths to support multi-modal inputs for the same volume.

**Transforms** (`torch_em/transform/`) are the mechanism that turns instance labels into training targets. Key ones: `BoundaryTransform`, `AffinityTransform`, `NoToBackgroundBoundaryTransform`, distance-based transforms. Augmentations (`get_augmentations`) are GPU/CPU differentiable via kornia.

**Models** (`torch_em/model/`): `UNet2d`, `UNet3d`, `AnisotropicUNet`, `UNETR`, `ProbabilisticUNet`, vision transformers, ViM/U-Net.

**Losses** (`torch_em/loss/`): `DiceLoss` is the default. Also Dice, contrastive/embedding (`ContrastiveLoss`, `SPOCOLoss` — see `EMBEDDING_LOSSES`), clDice, distance-based, and `LossWrapper`/`CombinedLoss` composition helpers.

**bioimage.io export.** Trained checkpoints export to the bioimage.io model format via `torch_em.util.export_bioimageio_model` (in `util/modelzoo.py`) for deployment in ilastik/deepImageJ. NOTE: `modelzoo` is deliberately NOT imported in `util/__init__.py` — it pulls in the heavy `bioimageio.core`, so `import torch_em` stays light and works without it. Import modelzoo functions explicitly when needed.

**Specialized training submodules** each ship their own trainer/loader: `self_training/` (mean-teacher, FixMatch unsupervised), `shallow2deep/` (Matskevych et al.), `classification/`. `multi_gpu_training.py` handles distributed training (the trainer's `rank` arg).

## Datasets (`torch_em/data/datasets/`)

Organized into four domains: `electron_microscopy/`, `light_microscopy/`, `histopathology/`, `medical/` (all star-imported into the package namespace). There are hundreds; new ones are added frequently.

Every dataset module follows the same four-function convention — match it exactly when adding a new dataset:
- `get_<name>_data(path, ..., download)` — downloads/prepares raw data on disk. If download is impossible (license, registration), raise an error explaining the manual steps.
- `get_<name>_paths(...)` — returns the filepaths to the prepared inputs.
- `get_<name>_dataset(...)` — returns the PyTorch `Dataset`.
- `get_<name>_loader(...)` — returns the `DataLoader`.

Each module declares `URLS` and `CHECKSUMS` constants. Use the shared helpers in `data/datasets/util.py`: `download_source`, `download_source_gdrive`/`_empiar`/`_kaggle`/`_tcia`/`_synapse`, `unzip`, `update_kwargs`. `scripts/datasets/` holds visualization/check scripts for each dataset.

## CLI

Console entry points (defined in `setup.py`, implemented in `cli.py` and `util/`): `torch_em.train_2d_unet`, `torch_em.train_3d_unet`, `torch_em.predict`, `torch_em.predict_with_tiling`, `torch_em.export_bioimageio_model`, `torch_em.validate_checkpoint`.
