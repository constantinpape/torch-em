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
- No callback logic; to extend the core functionality inherit from `torch_em.trainer.DefaultTrainer` instead.
- All data-loading is lazy to support training on large datasets.

`torch_em` can be installed via conda: `conda install -c conda-forge torch_em`.
Find an example script for how to train a 2D U-Net with it below and check out the [documentation](https://constantinpape.github.io/torch-em/torch_em.html) for more details.

```python
# Train a 2d U-Net for foreground and boundary segmentation of nuclei, using data from
# https://github.com/mpicbg-csbd/stardist/releases/download/0.1.0/dsb2018.zip

import torch_em
from torch_em.model import UNet2d
from torch_em.data.datasets import get_dsb_loader

model = UNet2d(in_channels=1, out_channels=2)

# Transform to convert from instance segmentation labels to foreground and boundary probabilties.
label_transform = torch_em.transform.BoundaryTransform(add_binary_target=True, ndim=2)

# Create the training and validation data loader.
data_path = "./dsb"  # The training data will be downloaded and saved here.
train_loader = get_dsb_loader(
    data_path, 
    patch_shape=(1, 256, 256),
    batch_size=8,
    split="train",
    download=True,
    label_transform=label_transform,
)
val_loader = get_dsb_loader(
    data_path, 
    patch_shape=(1, 256, 256),
    batch_size=8,
    split="test",
    label_transform=label_transform,
)

# The trainer handles the details of the training process.
# It will save checkpoints in "checkpoints/dsb-boundary-model"
# and the tensorboard logs in "logs/dsb-boundary-model".
trainer = torch_em.default_segmentation_trainer(
    name="dsb-boundary-model",
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    learning_rate=1e-4,
)
trainer.fit(iterations=5000)  # Fit for 5000 iterations.

# Export the trained model to the bioimage.io model format.
from glob import glob
import imageio
from torch_em.util import export_bioimageio_model

# Load one of the images to use as reference image.
# Crop it to a shape that is guaranteed to fit the network.
test_im = imageio.imread(glob(f"{data_path}/test/images/*.tif")[0])[:256, :256]

# Export the model.
export_bioimageio_model("./checkpoints/dsb-boundary-model", "./bioimageio-model", test_im)
```
