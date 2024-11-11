# ViM-UNet: Vision Mamba in Biomedical Segmentation

We introduce **ViM-UNet**, a novel segmentation architecture based on [Vision Mamba](https://github.com/hustvl/Vim) for instance segmentation in microscopy.

This is the documentation for the installation instructions, known issues and linked suggestions and benchmarking scripts.

## TLDR
1. Please install [`torch-em`](https://github.com/constantinpape/torch-em) and `ViM` (based on our fork: https://github.com/anwai98/Vim)
2. Supports `ViM Tiny` and `ViM Small` for 2d segmentation using ViM-UNet.
3. Check out [our preprint](https://arxiv.org/abs/2404.07705) (accepted at [MIDL 2024 - Short Paper](https://openreview.net/forum?id=PYNwysgFeP)) for more details.
    - The key observation: "ViM-UNet performs similarly or better that UNet (depending on the task), and outperforms UNETR while being more efficient." Its main advantage is for segmentation problems that rely on larger context.

## Benchmarking Methods

### Re-implemented methods in `torch-em`:
1. [ViM-UNet](https://github.com/constantinpape/torch-em/blob/main/torch_em/model/vim.py)
2. [UNet](https://github.com/constantinpape/torch-em/blob/main/torch_em/model/unet.py)
3. [UNETR](https://github.com/constantinpape/torch-em/blob/main/torch_em/model/unetr.py)

### External methods:

> [Here](https://github.com/computational-cell-analytics/vimunet-benchmarking) are the scripts to run the benchmarking for the reference methods.

1. nnU-Net (see [here](https://github.com/MIC-DKFZ/nnUNet) for installation instructions)
2. U-Mamba (see [here](https://github.com/bowang-lab/U-Mamba#installation) for installation instructions, and [issues]() encountered with our suggestions to take care of them)

## Installation

### For ViM-UNet:
1. Create a new environment and activate it:
```bash
$ mamba create -n vimunet python=3.10.13
$ mamba activate vimunet
```
2. Install `torch-em` [from source](https://github.com/constantinpape/torch-em#from-source).

3. Install `PyTorch`:
```bash
$ pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
```
> Q1. Why use `pip`? - for installation consistency

> Q2. Why choose CUDA 11.8? - Vim seems to prefer $\le$ 11.8 ([see here](https://github.com/hustvl/Vim/issues/51))

4. Install `ViM` and related dependencies (`causal-conv1d`\**, `mamba`, `Vim`\***):
```bash
$ git clone https://github.com/anwai98/Vim.git
$ cd Vim
$ git checkout dev  # Our latest changes are hosted at 'dev'.
$ pip install -r vim/vim_requirements.txt
$ pip install -e causal-conv1d
$ pip install -e mamba-1p1p1
$ pip install -e .
```

<!-- elf, kornia_rs, kornia, natsort, tensorboard -->
<!-- export LD_LIBRARY_PATH=/scratch/usr/nimanwai/micromamba/envs/vimunet/lib/ -->
<!-- pip install causal_conv1d==1.1.1 -->

> NOTE: The installation is sometimes a bit tricky, but following the steps and keeping the footnotes in mind should do the trick.
> We are working on providing an easier and more stable installation, [see this issue](https://github.com/constantinpape/torch-em/issues/237).

### For UNet and UNETR

1. Install `torch-em` [from source](https://github.com/constantinpape/torch-em#from-source).
2. Install `segment-anything` [from source](https://github.com/facebookresearch/segment-anything#installation).


## Known Issues and Suggestions
- `GLIBCXX_<VERSION>` related issues:
    - Suggestion: Specify your path to the mamba environment to `LD_LIBRARY_PATH`. For example,
    ```bash
    $ export LD_LIBRARY_PATH=/scratch/usr/nimanwai/mambaforge/lib/
    ```

- `FileNotFoundError: [Error 2] No such file or directory: 'ldconfig'`:
    - Suggestion: Possible reason is that the path variable isn't set correctly. I found this [here](https://unix.stackexchange.com/questions/160019/dpkg-cannot-find-ldconfig-start-stop-daemon-in-the-path-variable) quite useful. You can provide it as the following example:
    ```bash
    $ export PATH=$PATH:/usr/sbin  # it could also be located at /usr/bin, etc. please check your system configurations for this.
    ```

- **`NameError: name 'bare_metal_version' is not defined` while installing `causal-conv1d`:
    - Suggestion: This one's a bit tricky. From our findings, the possible issue is that the path to `CUDA_HOME` isn't visible to the installed PyTorch. The quickest way to test this is: `python -c "from torch.utils.cpp_extension import CUDA_HOME; print(CUDA_HOME)"`. It's often stored at `/usr/local/cuda`, hence to expose the path, here's the example script: `export CUDA_HOME=/usr/local/cuda`.
        > NOTE: If you are using your cluster's cuda installation and not sure where is it located, this should do the trick: `module show cuda/$VERSION`

- ***Remember to install the suggested `ViM` branch for installation. It's important as we enable a few changes to: a) automatically install the vision mamba as a developer module, and b) setting AMP to false for known issues (see [mention 1](https://github.com/hustvl/Vim/issues/30) and [mention 2](https://github.com/bowang-lab/U-Mamba/issues/8) for hints)
