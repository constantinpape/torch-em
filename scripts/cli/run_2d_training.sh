#! /bin/bash 

# Example commaind for using the CLI to run training of a 2D U-Net.
torch_em.train_2d_unet\
    -i data/isbi.h5 -k raw\
    -l data/isbi.h5 --training_label_key labels/gt_segmentation\
    -b 1 -p 1 128 128 -m boundaries --name 2d-unet
