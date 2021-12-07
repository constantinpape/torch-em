# LivCell Model

Training a model for instance segmentation of cells in phase-contrast microscopy images:
- `train_affinities.py`: train a model that predicts foreground and affinities (for postprocessing with Mutex Watershed)
- `train_boundaries.py`: train a model that predicts foreground and boundaries (for postprocessing e.g. with connected components)
- `export_bioimageio_model.py`: export trained model in the bioimage.io model format
- `validate_model.py`: run prediction and segmentation with a model in bioimage.io format and save or validate the results

Two models trained on [data from the LiveCell publication](https://doi.org/10.1038/s41592-021-01249-6) are available on bioimage.io:
- [affinity model](TODO)
- [boundary model](TODO)
