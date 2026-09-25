## UNETR
## Integrating SegmentAnything's Vision Transformer

The UNETR is implemented by adapting the vision transformer from Segment Anything for biomedical image segmentation.

Key Mentions:
- It's expected to install [SegmentAnything](https://github.com/facebookresearch/segment-anything) for this.
- The supported models are ViT Base, ViT Large and ViT Huge. They are often abbreviated as: [`vit_b`/`vit_l`/`vit_h`]
- The advantage of using SegmentAnything's vision transformer is to enable loading the pretrained weights without any hassle. It's exposed in the `UNETR` class configuration under the argument name: `encoder_checkpoint_path` - You need to pass the checkpoints from the pretrained [SegmentAnything models](https://github.com/facebookresearch/segment-anything#model-checkpoints) to initialize the SAM weights to the (ViT) encoder backbone (click on the model names to download them - [vit_b](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) / [vit_l](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth) / [vit_h](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth))


### How to train UNETR from scratch?
```python
from torch_em.model import UNETR
model = UNETR(
    encoder=<MODEL_NAME>,  # name of the vit backbone (see supported abbreviations above)
    out_channels=<OUTPUT_CHANNELS>,  # number of output channels matching the segmentation targets
)
```

### How to train UNETR, initialized with pretrained SegmentAnything weights?
```python
from torch_em.model import UNETR
unetr = UNETR(
    encoder=<MODEL_NAME>,  # name of the vit backbone (see supported abbreviations above)
    out_channels=<OUTPUT_CHANNELS>,  # number of output channels matching the segmentation targets
    encoder_checkpoint_path=<PATH_TO_CHECKPOINT>,  # path to the pretrained model weights
    use_sam_stats=True  # uses the image statistics from SA-1B dataset
)
```

## Description:
- `for_vimunet_benchmarking/`: (see [ViM-UNet description](https://github.com/constantinpape/torch-em/blob/main/vimunet.md) for details)
    - `run_livecell.py`: Benchmarking UNet and UNETR for cell segmentation in phase contrast microscopy.
    - `run_cremi.py`: Benchmarking UNet and UNETR for neurites segmentation in electron microscopy.

### Additional Experiments:
- `dsb/`: Experiments on DSB data for segmentation of nuclei in light microscopy.
- `livecell/`: Experiments on LIVECell data for segmentation of cells in phase contrast microscopy.
