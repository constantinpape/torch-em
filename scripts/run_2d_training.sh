#! /bin/bash 

# Example commaind for using the CLI to run training of a 2D U-Net.
# TODO Use some proper data that can be downloaded from zenodo or so

torch_em.train_2d_unet -i /home/pape/Work/data/isbi2012/vnc_train_volume.h5\
    -k volumes/raw\
    -l /home/pape/Work/data/isbi2012/vnc_train_volume.h5\
    --training_label_key volumes/labels/neuron_ids_3d\
    -b 1\
    -p 1 128 128\
    -m boundaries
