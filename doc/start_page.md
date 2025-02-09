[torch_em](https://github.com/constantinpape/torch-em) is a library for deep learning in microscopy images.
It supports segmentation and other relevant image analysis tasks. It is based on [PyTorch](https://pytorch.org/).

# Installation

## From conda

You can use `conda` (or its faster alternative [mamba](https://mamba.readthedocs.io/en/latest/)) to install `torch_em` and its dependencies from the `conda-forge` package:
```bash
conda install -c conda-forge torch_em
```
This command should work on Linux and Mac OS, on windows the installation is a bit more complex:
```bash
conda install -c pytorch -c nvidia -c conda-forge torch_em "nifty=1.2.1=*_4" "protobuf <5"
```

## From source

To install `torch_em` from source you should create a dedicated conda environment.
We provide the environment file `environment.yaml` for this. After cloning the `torch_em` repository,
you can set it up and then install `torch_em` via:
```bash
conda env create -f environment.yaml
conda activate torch-em-dev
pip install -e .
```
This should work on Linux and Mac OS. On windows you have to use a different environment file:
`environment_cpu_win.yaml` for setting up an environment with a PyTorch version with CPU support or
`environment_gpu_win.yaml` for a PyTorch version with GPU support.

# Usage & Examples

`torch_em` provides functionality for training deep neural networks, for segmentation tasks in `torch_em.segmentation`
and classification tasks in `torch_em.classification`.
To customize the models and model training it implements multiple neural network architectures in `torch_em.model`,
loss functions in `torch_em.loss`, data pipelines in `torch_em.data`, and data transformations in `torch_em.transform`.
It also provides ready-to-use datasets for many different bio-medical image analysis tasks in `torch_em.data.datasets`.
These datasets are explained in detail in [Biomedical Datasets](#biomedical-datasets).
It provides inference functionality for neural networks in `torch_em.util.prediction` and a function to export trained networks to [BioImage.IO](https://bioimage.io/#/) in `torch_em.util.modelzoo`.

You can find the scripts for training [2d U-Nets](https://doi.org/10.1007/978-3-319-24574-4_28) and [3d U-Nets](https://doi.org/10.1007/978-3-319-46723-8_49) for various segmentation tasks in [experiments/unet-segmentation](https://github.com/constantinpape/torch-em/tree/main/experiments/unet-segmentation). We also provide two example notebooks that demonstrate U-Net training in more detail:
- [2D-UNet Example Notebook](https://github.com/constantinpape/torch-em/blob/main/experiments/2D-UNet-Training.ipynb) to train a 2d UNet for a segmentation task. [Available on google colab](https://colab.research.google.com/github/constantinpape/torch-em/blob/main/experiments/2D-UNet-Training.ipynb).
- [3D-UNet Example Notebook](https://github.com/constantinpape/torch-em/blob/main/experiments/3D-UNet-Training.ipynb) to train a 3d UNet for a segmentation task. [Available on google colab](https://colab.research.google.com/github/constantinpape/torch-em/blob/main/experiments/3D-UNet-Training.ipynb).

## Advanced Features

`torch_em` also implements multiple advanced network architectures and training algorithms:
- Semi-supervised training or domain adaptation via [FixMatch](https://arxiv.org/abs/2001.07685) and [MeanTeacher](https://arxiv.org/abs/1703.01780).
- Random forest based domain adaptation from [Shallow2Deep](https://doi.org/10.1101/2021.11.09.467925).
    - This functionality is implemented in `torch_em.shallow2deep`.
    - Scripts for training Shallow2Deep models are located in [experiments/shallow2deep](https://github.com/constantinpape/torch-em/tree/main/experiments/shallow2deep).
- Training models for embedding prediction with sparse instance labels with [SPOCO](https://arxiv.org/abs/2103.14572).
    - This functionality is implemented in `torch_em.trainer.spoco_trainer`.
    - Scripts for training models with SPOCO are located in [experiments/spoco](https://github.com/constantinpape/torch-em/tree/main/experiments/spoco).
- Transformer-based segmentation via [UNETR](https://doi.org/10.48550/arXiv.2103.10504), with a choice of vision transformer backbones from [Segment Anything](https://doi.org/10.48550/arXiv.2304.02643) or [Masked Autoencoder](https://doi.org/10.48550/arXiv.2111.06377).
    - This functionality is implemented in `torch_em.model.unetr`.
    - Scripts for training UNETR models are located in [experiments/vision-transformer/unetr](https://github.com/constantinpape/torch-em/tree/main/experiments/vision-transformer/unetr).
- [Mamba](https://arxiv.org/abs/2312.00752)-based segmentation via [ViM-UNet](https://doi.org/10.48550/arXiv.2404.07705).
    - This functionality is implemented in `torch_em.model.vim`.
    - Scripts for training ViM-UNet models are located in [experiments/vision-mamba/vimunet](https://github.com/constantinpape/torch-em/tree/main/experiments/vision-mamba).

`torch_em` also enables data parallel multi-gpu training via `torch.distributed`. This functionality is implemented in `torch_em.multi_gpu_training`. See [scripts/run_multi_gpu_train.py](https://github.com/constantinpape/torch-em/blob/main/scripts/run_multi_gpu_train.py) for an example script.

## Command Line Interface

`torch_em` provides the following command line scripts:
- `torch_em.train_unet_2d` to train a 2D U-Net. 
- `torch_em.train_unet_3d` to train a 3D U-Net. 
- `torch_em.predict` to run prediction with a trained model.
- `torch_em.predict_with_tiling`to run prediction with tiling.
- `torch_em.export_bioimageio_model` to export a model to the modelzoo format.
- `torch_em.validate_checkpoint` to evaluate a model from a trainer checkpoint.

For more details run `<COMMAND> -h` for any of these commands.
The folder [scripts/cli](https://github.com/constantinpape/torch-em/tree/main/scripts/cli) contains some examples for how to use the CLI.

## Projects using `torch_em`

Multiple research projects are built with `torch_em`:
- [Probabilistic Domain Adaptation for Biomedical Image Segmentation](https://doi.org/10.48550/arXiv.2303.11790) | [Code Repository](https://github.com/computational-cell-analytics/Probabilistic-Domain-Adaptation)
- [Segment Anything for Microscopy](https://doi.org/10.1101/2023.08.21.554208) | [Code Repository](https://github.com/computational-cell-analytics/micro-sam)
- [ViM-UNet: Vision Mamba for Biomedical Segmentation](https://doi.org/10.48550/arXiv.2404.07705) | [Code Repository](https://github.com/constantinpape/torch-em/blob/main/vimunet.md)
- [SynapseNet: Deep Learning for Automatic Synapse Reconstruction](https://doi.org/10.1101/2024.12.02.626387) | [Code Repository](https://github.com/computational-cell-analytics/synapse-net)
- [MedicoSAM: Towards foundation models for medical image segmentation](https://arxiv.org/abs/2501.11734) | [Code Repository](https://github.com/computational-cell-analytics/medico-sam)
- [Parameter Efficient Fine-Tuning of Segment Anything Model](https://arxiv.org/abs/2502.00418) | [Code Repository](https://github.com/computational-cell-analytics/peft-sam)
- [Segment Anything for Histopathology](https://arxiv.org/abs/2502.00408) | [Code Repository](https://github.com/computational-cell-analytics/patho-sam)
