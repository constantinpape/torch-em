# ViM-UNet: Vision Mamba in Biomedical Segmentation

Experiments for ViM-UNet. Check the [ViM-UNet documentation](https://github.com/constantinpape/torch-em/blob/main/vimunet.md) for details on the method and installation instructions.

Here are the experiments for instance segmentation on:
1. LIVECell for cell segmentation in phase-contrast microscopy.
    - You can run the boundary-based /distance-based experiments. See `run_livecell.py -h` for details.
    ```python
    python run_livecell.py -i <PATH_TO_DATA>
                           -s <PATH_TO_SAVE_CHECKPOINTS>
                           -m <MODEL_NAME>  # the supported models are 'vim_t', 'vim_s' and 'vim_b'
                           --train  # for training
                           --predict  # for inference on trained models
                           --result_path  <PATH_TO_SAVE_RESULTS>
                           # below is how you can provide the choice for training for either methods
                           --boundaries / --distances
    ```

2. CREMI for neurites segmentation in electron microscopy.
    - You can run the boundary-based experiment. See `run_livecell.py -h` for details. Below is an example script:
    ```python
    python run_cremi.py -i <PATH_TO_DATA>
                        -s <PATH_TO_SAVE_CHECKPOINTS>
                        -m <MODEL_NAME>  # the supported models are 'vim_t', 'vim_s' and 'vim_b'
                        --train  # for training
                        --predict  # for inference on trained models
                        --result_path  <PATH_TO_SAVE_RESULTS>
    ```

