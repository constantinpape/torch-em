# installation from https://github.com/hustvl/Vim
# encoder from https://github.com/hustvl/Vim
# decoder from https://github.com/constantinpape/torch-em

# pretrained model weights: vim_t - https://huggingface.co/hustvl/Vim-tiny/blob/main/vim_tiny_73p1.pth

import torch

from torch_em.model import UNETR

from vim.models_mamba import VisionMamba, rms_norm_fn, RMSNorm, layer_norm_fn

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

    def forward_features(self, x, inference_params=None):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        x = self.patch_embed(x)
        if self.if_cls_token:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_token, x), dim=1)

        if self.if_abs_pos_embed:
            x = x + self.pos_embed
            x = self.pos_drop(x)

        # mamba impl
        residual = None
        hidden_states = x
        for layer in self.layers:
            # rope about
            if self.if_rope:
                hidden_states = self.rope(hidden_states)
                if residual is not None and self.if_rope_residual:
                    residual = self.rope(residual)

            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        if self.final_pool_type == 'none':
            return hidden_states[:, -1, :]
        elif self.final_pool_type == 'mean':
            return hidden_states.mean(dim=1)
        elif self.final_pool_type == 'max':
            return hidden_states.max(dim=1)
        elif self.final_pool_type == 'all':
            return hidden_states
        else:
            raise NotImplementedError

    def forward(self, x, inference_params=None):
        x = self.forward_features(x, inference_params)

        if self.if_cls_token:  # remove the class token
            x = x[:, 1:, :]

        # let's get the patches back from the 1d tokens
        x = self.convert_to_expected_dim(x)

        return x  # from here, the tokens can be upsampled easily (N x H x W x C)


def get_vimunet_model(out_channels, device=None, checkpoint=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = ViM(
        img_size=1024,
        patch_size=16,
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

    model_state = None
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu")

        if checkpoint.endswith(".pth"):  # from Vim
            encoder_state = state["model"]
            encoder.load_state_dict(encoder_state)

        else:  # from torch_em
            model_state = state["model_state"]

    encoder.img_size = encoder.patch_embed.img_size[0]

    model = UNETR(
        encoder=encoder,
        out_channels=out_channels,
        resize_input=False,
        use_skip_connection=False,
        final_activation="Sigmoid"
    )

    if model_state is not None:
        model.load_state_dict(model_state)

    model.to(device)

    return model
