#! /bin/bash 

# Example commaind for using the CLI to run training of a 2D U-Net.
torch_em.train_3d_unet\
    -i data/isbi.h5 -k raw\
    -l data/isbi.h5 --training_label_key labels/gt_segmentation\
    -b 1 -p 8 96 96 -s [[1,2,2],[1,2,2],[2,2,2]] -m boundaries --name 3d-unet
