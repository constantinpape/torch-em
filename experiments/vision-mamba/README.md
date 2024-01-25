## Installation:

- `mamba create -n vision-mamba python=3.10.13`

(we stick to `pip` for installation consistency)

- Install PyTorch: `pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118`
- `git clone https://github.com/anwai98/Vim.git`
- `cd Vim`
- Install from `vim_requirements.txt` -> `pip install -r vim/vim_requirements.txt`
- `pip install -e causal_conv1d/`
- `pip install -e mamba/`


### Known Errors:

- `GLIBCXX_<VERSION>` related issues

Fix: `export LD_LIBRARY_PATH=/scratch/usr/nimanwai/mambaforge/lib/`

- `FileNotFoundError: [Errno 2] No such file or directory: 'ldconfig'`

Fix: `export PATH=$PATH:/usr/sbin`
