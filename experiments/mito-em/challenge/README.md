# MitoEM

Experiments for the [MitoEM Challenge](https://mitoem.grand-challenge.org/).

The scripts `train_affinities.py` / `train_boundaries.py` in the top-level folder are used to train an affinity model / boundary model for mitochondria instance segmentation.
They take different arguments for training models of different size and training / validating on different splits of the data.
The script `segment_and_submit.py` performs instance segmentation for the test volumes based on the affinity / boundary predictions and creates the segmentation in submission format.

There are some additional scripts for data preperation and visualisation:
- `check_result.py`: view cutout of results with [napari](https://github.com/napari/napari) for quick visual inspection
- `create_mobie_project.py`: create a [MoBIE](https://github.com/mobie/mobie) project based on the MitoEM data for data exploration and systematic comparison of results on the validation set
- `prepare_train_data.py`: prepare the data format expected by the training scripts
- `segment_and_validate.py`: run instance segmentation and validation for the validation volumes
- `segmentation_impl.py`: implementation of instance segmentation methods 

In order to set-up an environment that can run all scripts, follow [these installation instructions](https://github.com/constantinpape/torch-em) and install
additional requirements via
```
conda install -c conda-forge -c cpape napari mobie_utils
```
Then, install the MitoEM validation scripts from: https://github.com/ygCoconut/mAP_3Dvolume.git

## Results

The experiments for the submissions in the first round of the challenge were run with commit [101e406](https://github.com/constantinpape/torch-em/commit/101e406900d6cfb3451448b1b16f2b49eaa0a7f4)
