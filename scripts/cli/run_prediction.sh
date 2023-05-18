#! /bin/bash 

# Example commaind for using the CLI to run training of a 2D U-Net.

torch_em.predict\
    -c checkpoints/2d-unet -i data/input.tif -o data/pred2d.tif
