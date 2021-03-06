[![Build Status](https://github.com/constantinpape/torch-em/workflows/test/badge.svg)](https://github.com/constantinpape/torch-em/actions)

# Torch'em

Deep-learning based semantic and instance segmentation for 3D Electron Microscopy and other bioimage analysis problems based on pytorch.

This library is in a very early state, so don't expect a stable API, no bugs and there be dragons ;). Early feedback is highly appreciated, just open an issue!

Highlights:
- Functional API with sensible defaults to train a state-of-the-art segmentation model with a few lines of code.
- Differentiable augmentations on GPU and CPU thanks to [kornia](https://github.com/kornia/kornia).
- Off-the-shelf logging with [tensorboard](https://www.tensorflow.org/tensorboard) or [wandb](https://wandb.ai/site).
- Export trained models to [bioimage.io](https://bioimage.io/#/) model format with one function call.

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

model = UNet2d(in_channels=1, out_channels=2)

# transform to go from instance segmentation labels
# to foreground/background and boundary channel
label_transform = torch_em.transform.BoundaryTransform(
    add_binary_target=True, ndim=2
)

# training and validation data loader
train_loader = torch_em.default_segmentation_loader(
    "dsb2018/train/images", "*.tif",
    "dsb2018/train/masks", "*.tif",
    batch_size=8, patch_shape=(1, 256, 256),
    label_transform=label_transform,
    n_samples=250
)
val_loader = torch_em.default_segmentation_loader(
    "dsb2018/test/images", "*.tif",  # misusing the test data for validation ;)
    "dsb2018/test/masks", "*.tif",
    batch_size=8, patch_shape=(1, 256, 256),
    label_transform=label_transform,
    n_samples=25
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
test_im = imageio.imread(glob("dsb2018/test/images/*.tif")[0])

export_bioimageio_model("./checkpoints/dsb-boundary-model", test_im, "./bioimageio-model")
```

Check out [expirements/platynereis/mitochondria/train_affinities.py](https://github.com/constantinpape/torch-em/blob/main/experiments/platynereis/mitochondria/train_affinities.py) for a more advanced example.


## Command Line Scripts

TODO


## Installation

Two conda environment files are provided: `environment_cpu.yaml` for a pure cpu set-up and `environment_gpu.yaml` for a gpu set-up.
If you want to use the gpu version, make sure to set the correct cuda version for your system in the environment file.

You can set up a conda environment with all necessary dependencies like this:
```sh
conda env create -f <ENV>.yaml -n <ENV_NAME>
conda activate <ENV_NAME>
pip install -e .
```
