# Experiments

Training and evaluation of neural networks for biomedical image analysis with `torch_em`.
The subfolders `unet_segmentation`, `shallow2deep`, `spoco` and `probabilistic_domain_adaptation` contain code for different methods.

The best entrypoints for training a model yourself are the notebooks:
- `2D-UNet-Training`: train a 2d UNet for segmentation tasks, [available on google colab](https://colab.research.google.com/github/constantinpape/torch-em/blob/main/experiments/2D-UNet-Training.ipynb).
- `3D-UNet-Training`: train a 3d UNet for segmentation tasks, [available on google colab](https://colab.research.google.com/github/constantinpape/torch-em/blob/main/experiments/3D-UNet-Training.ipynb).

## unet_segmentation

This folder contains several experiments for training 2d or 3d U-Nets () for segmentation tasks.
Most of these models are available on [BioImage.IO](https://bioimage.io/#/).

Note: some of these experiments are based on older versions of `torch_em` and might not work due to (small) changes in function signatures.
If you encounter an issue with one of these experiments please open an issue!

## shallow2deep

Experiments for the re-implementation of [From Shallow to Deep: Exploiting Feature-Based Classifiers for Domain Adaptation in Semantic Segmentation](https://doi.org/10.3389/fcomp.2022.805166). The code here was used to train the models for the [ilastik Trainable Domain Adaptation Workflow](https://www.ilastik.org/documentation/tda/tda).


## spoco

Experiments for the re-implementation of [Sparse Object-Level Supervision for Instance Segmentation With Pixel Embeddings](https://openaccess.thecvf.com/content/CVPR2022/html/Wolny_Sparse_Object-Level_Supervision_for_Instance_Segmentation_With_Pixel_Embeddings_CVPR_2022_paper.html). Work in progress.


## probabilistic_domain_adaptation

Experiments for the re-implementation of [Probabilistic Domain Adaptation for Biomedical Image Segmentation](https://arxiv.org/abs/2303.11790). Work in progress.
