# CovidIF Model

Training a model for instance segmentation of cells in immunofluorescence microscopy images:
- `train_affinities.py`: train a model that predicts foreground and affinities (for postprocessing with Mutex Watershed)
- `train_boundaries.py`: train a model that predicts foreground and boundaries (for postprocessing seeded watershed, using additional nucleus segmentation)
- `export_bioimageio_model.py`: export trained model in the bioimage.io model format
- `validate_model.py`: run prediction and segmentation with a model in bioimage.io format and save or validate the results

Two models trained on [Vero E6 cells imaged in IF](https://zenodo.org/record/5092850) are available on bioimage.io:
- [affinity model](TODO)
- [boundary model](TODO)
