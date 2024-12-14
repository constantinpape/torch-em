[![DOC](https://shields.mitmproxy.org/badge/docs-pdoc.dev-brightgreen.svg)](https://constantinpape.github.io/torch-em/torch_em.html)
[![Build Status](https://github.com/constantinpape/torch-em/workflows/test/badge.svg)](https://github.com/constantinpape/torch-em/actions)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5108853.svg)](https://doi.org/10.5281/zenodo.5108853)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/torch_em/badges/version.svg)](https://anaconda.org/conda-forge/torch_em)

# torch-em

Deep-learning based semantic and instance segmentation for 3D Electron Microscopy and other bioimage analysis problems based on PyTorch.
Any feedback is highly appreciated, just open an issue!

Highlights:
- Functional API with sensible defaults to train a state-of-the-art segmentation model with a few lines of code.
- Differentiable augmentations on GPU and CPU thanks to [kornia](https://github.com/kornia/kornia).
- Off-the-shelf logging with [tensorboard](https://www.tensorflow.org/tensorboard) or [wandb](https://wandb.ai/site).
- Export trained models to [bioimage.io](https://bioimage.io/#/) model format with one function call to deploy them in [ilastik](https://www.ilastik.org/documentation/nn/nn) or [deepimageJ](https://deepimagej.github.io/deepimagej/).

Design:
- All parameters are specified in code, no configuration files.
- No callback logic; to extend the core functionality inherit from `trainer.DefaultTrainer` instead.
- All data-loading is lazy to support training on large data-sets.

```python
# train a 2d U-Net for foreground and boundary segmentation of nuclei
# using data from https://github.com/mpicbg-csbd/stardist/releases/download/0.1.0/dsb2018.zip

import torch
import torch_em
from torch_em.model import UNet2d
from torch_em.data.datasets import get_dsb_loader

model = UNet2d(in_channels=1, out_channels=2)

# transform to go from instance segmentation labels
# to foreground/background and boundary channel
label_transform = torch_em.transform.BoundaryTransform(
    add_binary_target=True, ndim=2
)

# training and validation data loader
data_path = "./dsb"  # the training data will be downloaded and saved here
train_loader = get_dsb_loader(
    data_path, 
    patch_shape=(1, 256, 256),
    batch_size=8,
    split="train",
    download=True,
    label_transform=label_transform
)
val_loader = get_dsb_loader(
    data_path, 
    patch_shape=(1, 256, 256),
    batch_size=8,
    split="test",
    label_transform=label_transform
)

# the trainer object that handles the training details
# the model checkpoints will be saved in "checkpoints/dsb-boundary-model"
# the tensorboard logs will be saved in "logs/dsb-boundary-model"
trainer = torch_em.default_segmentation_trainer(
    name="dsb-boundary-model",
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    learning_rate=1e-4,
    device=torch.device("cuda")
)
trainer.fit(iterations=5000)

# export bioimage.io model format
from glob import glob
import imageio
from torch_em.util import export_bioimageio_model

# load one of the images to use as reference image image
# and crop it to a shape that is guaranteed to fit the network
test_im = imageio.imread(glob(f"{data_path}/test/images/*.tif")[0])[:256, :256]

export_bioimageio_model("./checkpoints/dsb-boundary-model", "./bioimageio-model", test_im)
```

For a more in-depth example, check out one of the example notebooks:
- [2D-UNet](https://github.com/constantinpape/torch-em/blob/main/experiments/2D-UNet-Training.ipynb): train a 2d UNet for a segmentation task. [Available on google colab](https://colab.research.google.com/github/constantinpape/torch-em/blob/main/experiments/2D-UNet-Training.ipynb).
- [3D-UNet](https://github.com/constantinpape/torch-em/blob/main/experiments/3D-UNet-Training.ipynb): train a 3d UNet for a segmentation task. [Available on google colab](https://colab.research.google.com/github/constantinpape/torch-em/blob/main/experiments/3D-UNet-Training.ipynb).


## Installation

### From mamba

[mamba](https://mamba.readthedocs.io/en/latest/) is a drop-in replacement for conda, but much faster. While the steps below may also work with `conda`, it's highly recommended using `mamba`. You can follow the instructions [here](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html) to install `mamba`.

You can install `torch_em` from conda-forge:
```
mamba install -c conda-forge torch_em
```
Please check out [pytorch.org](https://pytorch.org/) for more information on how to install a PyTorch version compatible with your system.

### From source

It's recommmended to set up a conda environment for using `torch_em`.
Two conda environment files are provided: `environment_cpu.yaml` for a pure CPU set-up and `environment_gpu.yaml` for a GPU set-up.
If you want to use the GPU version, make sure to set the correct CUDA version for your system in the environment file, by modifiying [this-line](https://github.com/constantinpape/torch-em/blob/main/environment_gpu.yaml#L9).

You can set up a conda environment using one of these files like this:
```bash
mamba env create -f <ENV>.yaml -n <ENV_NAME>
mamba activate <ENV_NAME>
pip install -e .
```
where `<ENV>.yaml` is either `environment_cpu.yaml` or `environment_gpu.yaml`.


## Features

- Training of [2d U-Nets](https://doi.org/10.1007/978-3-319-24574-4_28) and [3d U-Nets](https://doi.org/10.1007/978-3-319-46723-8_49) for various segmentation tasks.
- Random forest based domain adaptation from [Shallow2Deep](https://doi.org/10.1101/2021.11.09.467925)
- Training models for embedding prediction with sparse instance labels from [SPOCO](https://arxiv.org/abs/2103.14572)
- Training of [UNETR](https://doi.org/10.48550/arXiv.2103.10504) for various 2d segmentation tasks, with a flexible choice of vision transformer backbone from [Segment Anything](https://doi.org/10.48550/arXiv.2304.02643) or [Masked Autoencoder](https://doi.org/10.48550/arXiv.2111.06377).
- Training of [ViM-UNet](https://doi.org/10.48550/arXiv.2404.07705) for various 2d segmentation tasks.


## Command Line Scripts

A command line interface for training, prediction and conversion to the [bioimage.io modelzoo](https://bioimage.io/) format wll be installed with `torch_em`:
- `torch_em.train_unet_2d`: train a 2D U-Net. 
- `torch_em.train_unet_3d`: train a 3D U-Net. 
- `torch_em.predict`: run prediction with a trained model.
- `torch_em.predict_with_tiling`: run prediction with tiling.
- `torch_em.export_bioimageio_model`: export a model to the modelzoo format.

For more details run `<COMMAND> -h` for any of these commands.
The folder [scripts/cli](https://github.com/constantinpape/torch-em/tree/main/scripts/cli) contains some examples for how to use the CLI.

Note: this functionality was recently added and is not fully tested.

## Research Projects using `torch-em`

- [Probabilistic Domain Adaptation for Biomedical Image Segmentation](https://doi.org/10.48550/arXiv.2303.11790) | [Code Repository](https://github.com/computational-cell-analytics/Probabilistic-Domain-Adaptation)
- [Segment Anything for Microscopy](https://doi.org/10.1101/2023.08.21.554208) | [Code Repository](https://github.com/computational-cell-analytics/micro-sam)
- [ViM-UNet: Vision Mamba for Biomedical Segmentation](https://doi.org/10.48550/arXiv.2404.07705) | [Code Repository](https://github.com/constantinpape/torch-em/blob/main/vimunet.md)
- [SynapseNet: Deep Learning for Automatic Synapse Reconstruction](https://doi.org/10.1101/2024.12.02.626387) | [Code Repository](https://github.com/computational-cell-analytics/synapse-net)
