# **Vi**sion **M**amba-based **UNet** for Biomedical Image Segmentation **(ViMUNet)**

Extending Vision Mamba (Vim) for instance segmentation in microscopic images to effectively capture long-range dependencies with efficiency.

Paper link: TODO

Notebook link: TODO

## Installation:

> NOTE: The installation is a bit tricky, but following the steps should do the trick.

- Create a new mamba environment: `mamba create -n vimunet python=3.10.13`
- Activate the environment: `mamba activate vimunet`
- Install PyTorch: `pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118`
  - Why use `pip`? - For installation consistency
- Install Vim: `git clone https://github.com/anwai98/Vim.git`
- Enter the directory: `cd Vim`
- Install Vim-related dependencies: `pip install -r vim/vim_requirements.txt`
- Install causal-conv1d: `pip install -e causal_conv1d/`
- Install mamba: `pip install -e mamba/`
- Install Vim: `pip install -e .`

### Known Issues:
1. `GLIBCXX_<VERSION>` related issues
    - Suggestion: explicity mention your path to the conda environment to `LD_LIBRARY_PATH`. Example: `export LD_LIBRARY_PATH=/scratch/usr/nimanwai/mambaforge/lib/`
2. `FileNotFoundError: [Errno 2] No such file or directory: 'ldconfig'`
    - Suggestion: Possible reasons are that path variable isn't set correctly. Provide it as the following example: `export PATH=$PATH:/usr/sbin`
3. `NameError: name 'bare_metal_version' is not defined` while installing `causal-conv1d`
    - Suggestion: It's possible that the path to `CUDA_HOME` isn't visible to the installed PyTorch. The quickest way to test this is: `python -c "from torch.utils.cpp_extension import CUDA_HOME; print(CUDA_HOME)"` - if the code returns None, then you'd need to specify the path to `CUDA_HOME`. Often, it's stored at `/usr/local/cuda`, so the script would be `export CUDA_HOME=/usr/loca/cuda`. If you are using your cluster's cuda installation and not sure where is it located, this might do the trick: `module show cuda/$VERSION`.

If you use our work, please cite the following:

1. [Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model](https://arxiv.org/abs/2401.09417): Zhu et al.
