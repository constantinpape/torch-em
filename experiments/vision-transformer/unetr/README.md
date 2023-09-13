## SAM's ViT Initialization in UNETR

Note:
- `model_type` - [`vit_b`/`vit_l`/`vit_h`]
- `out_channels` - Number of output channels
- `encoder_checkpoint_path` - Pass the checkpoints from the pretrained [Segment Anything](https://github.com/facebookresearch/segment-anything) models to initialize the SAM weights to the (ViT) encoder backbone (Click on the model names to download them - [ViT-b](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) / [ViT-l](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth) / [ViT-h](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)) 


### How to initialize ViT models for UNETR?
```
from torch_em.model import UNETR
unetr = UNETR(encoder=model_type, out_channels=out_channels, encoder_checkpoint_path=checkpoint_path)
```

### Vanilla ViT models for UNETR
```
from torch_em.model import UNETR
unetr = UNETR(encoder=model_type, out_channels=out_channels)
```
