# DSB Model

Training a model for instance segmentation of nuclei in fluorescence light microscopy images:
- `train_affinities.py`: train a model that predicts foreground and affinities (for postprocessing with Mutex Watershed)
- `train_boundaries.py`: train a model that predicts foreground and boundaries (for postprocessing e.g. with connected components)
- `export_bioimageio_model.py`: export trained model in the bioimage.io model format
- `validate_model.py`: run prediction and segmentation with a model in bioimage.io format and save or validate the results

Two models trained on [data from the DSB nucleus segmentation challenge](https://github.com/stardist/stardist/releases/download/0.1.0/dsb2018.zip) are available on bioimage.io:
- [affinity model](TODO)
- [boundary model](TODO)
