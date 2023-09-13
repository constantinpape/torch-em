## Usage of SAM's ViT initialization in UNETR

### Initialize ViT models for UNETR
```
from torch_em.model import UNETR
unetr = UNETR(encoder=model_type, out_channels=out_channels, encoder_checkpoint_path=checkpoint_path)
```

### Vanilla ViT models for UNETR
```
from torch_em.model import UNETR
unetr = UNETR(encoder=model_type, out_channels=out_channels)
```