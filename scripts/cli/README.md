# Command Line Scripts

Example scripts for using the `torch_em` CLI:
- `download_test_data.py`: will download the data used in all the script. You need to run this before running any of the other scripts.
- `run_2d_training.sh`: example for training a 2D U-Net with the CLI. 
- `run_3d_training.sh`: example for training a 3D U-Net with the CLI.
- `run_prediction.sh`: example for running prediction with the CLI. You first need to run `run_2d_training`, because this script will use the checkpoint from training.
- `run_prediction_with_tiling.sh`: example for running prediction with tiling with the CLI. You first need to run `run_2d_training`, because this script will use the checkpoint from training.
- `export_to_bioimageio.sh`: example for exporting a trained model to the BioImage.IO model format with the CLI. You first need to run `run_2d_training`, because this script will use the checkpoint from training.
