from collections import OrderedDict

import torch
from torch_em.model import UNETR

checkpoint = "imagenet.pth"
encoder_state = torch.load(checkpoint, map_location="cpu")["model"]
encoder_state = OrderedDict({
    k: v for k, v in encoder_state.items()
    if (k != "mask_token" and not k.startswith("decoder"))
})

unetr_model = UNETR(backbone="mae", encoder="vit_l", encoder_checkpoint=encoder_state)
