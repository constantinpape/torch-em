#!/bin/sh
. $PREFIX/etc/profile.d/conda.sh  # do not edit
conda activate $PREFIX            # do not edit

# pip dependencies: install kornia via pip to avoid installing pytorch
pip install --no-deps kornia
