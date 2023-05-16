#! /bin/bash 

# Example commaind for using the CLI to run training of a 2D U-Net.
# TODO Use some proper data that can be downloaded from zenodo or so

torch_em.predict_with_tiling -c checkpoints/2d-unet-training-ab456022-f430-11ed-9fc1-9cb6d03fc2ca\
    -i /home/pape/Work/data/isbi2012/isbi2012_test_volume.h5\
    -k volumes/raw\
    -b 1 512 512\
    -o pred3d.tif
