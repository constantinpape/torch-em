#! /bin/bash 

# Example commaind for using the CLI to run training of a 3D U-Net.
torch_em.predict_with_tiling\
    -c checkpoints/2d-unet -i data/isbi.h5 -k raw\
    -b 1 512 512 -o data/pred3d.tif
