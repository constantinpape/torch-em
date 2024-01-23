# installation from https://github.com/hustvl/Vim
# encoder from https://github.com/hustvl/Vim
# decoder from https://github.com/constantinpape/torch-em

# pretrained model weights: vim_t - https://huggingface.co/hustvl/Vim-tiny/blob/main/vim_tiny_73p1.pth

import torch

from torch_em.model import UNETR

from vim.models_mamba import VisionMamba

from timm.models.vision_transformer import _cfg


class ViM(VisionMamba):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)

    def convert_to_expected_dim(self, inputs_):
        # reshape the outputs to desired shape (N x H*W X C -> N x H x W x C)
        rdim = inputs_.shape[1]
        dshape = int(rdim ** 0.5)  # finding the square root of the outputs for obtaining the patch shape
        inputs_ = torch.unflatten(inputs_, 1, (dshape, dshape))
        inputs_ = inputs_.permute(0, 3, 1, 2)
        return inputs_

    def forward(self, x, inference_params=None):
        x = self.forward_features(x, inference_params)

        # let's get the patches back from the 1d tokens
        x = self.convert_to_expected_dim(x)

        return x  # from here, the tokens can be upsampled easily (N x H x W x C)


def get_vimunet_model(device=None, checkpoint=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = VisionMamba(
        atch_size=16,
        embed_dim=192,
        depth=24,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        final_pool_type='all',
        if_abs_pos_embed=True,
        if_rope=True,
        if_rope_residual=True,
        bimamba_type="v2",
        if_cls_token=True,
    )

    encoder.default_cfg = _cfg()

    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu")
        encoder_state = state["model"]
        encoder.load_state_dict(encoder_state)

    encoder.img_size = encoder.patch_embed.img_size[0]

    model = UNETR(
        encoder=encoder,
        out_channels=1,
        resize_input=False,
        use_skip_connection=False,
        final_activation="Sigmoid"
    )
    model.to(device)

    return model
