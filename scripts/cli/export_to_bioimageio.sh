#! /bin/bash 

# Example command for using the CLI to export a trained model to the bioimage.io model format.

torch_em.export_bioimageio_model\
    -p checkpoints/2d-unet -d data/input.tif -f data/exported_model --name my-unet
