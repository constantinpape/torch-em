#! /bin/bash 

# Example commaind for using the CLI to run training of a 2D U-Net.
# TODO Use some proper data that can be downloaded from zenodo or so

torch_em.train_3d_unet -i /home/pape/Work/data/isbi2012/vnc_train_volume.h5\
    -k volumes/raw\
    -l /home/pape/Work/data/isbi2012/vnc_train_volume.h5\
    --training_label_key volumes/labels/neuron_ids_3d\
    -b 1\
    -p 8 96 96\
    -s [[1,2,2],[1,2,2],[2,2,2]]\
    -m boundaries
