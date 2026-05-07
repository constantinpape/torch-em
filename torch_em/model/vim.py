"""@private
"""

# installation from https://github.com/hustvl/Vim
# encoder from https://github.com/hustvl/Vim
# decoder from https://github.com/constantinpape/torch-em

# pretrained model weights: vim_t - https://huggingface.co/hustvl/Vim-tiny/blob/main/vim_tiny_73p1.pth

import random

import torch

from .unetr import UNETR

try:
    from vim.models_mamba import VisionMamba, rms_norm_fn, RMSNorm, layer_norm_fn
    _have_vim_installed = True
except ImportError:
    VisionMamba = object
    rms_norm_fn = RMSNorm = layer_norm_fn = None
    _have_vim_installed = False

try:
    from timm.models.vision_transformer import _cfg
except ImportError:
    _cfg = None


class ViM(VisionMamba):
    def __init__(self, **kwargs):
        assert _have_vim_installed, "Please install 'Vim'."
        super().__init__(**kwargs)

    def convert_to_expected_dim(self, inputs_):
        # reshape the outputs to desired shape (N x H*W X C -> N x H x W x C)
        rdim = inputs_.shape[1]
        dshape = int(rdim ** 0.5)  # finding the square root of the outputs for obtaining the patch shape
        inputs_ = torch.unflatten(inputs_, 1, (dshape, dshape))
        inputs_ = inputs_.permute(0, 3, 1, 2)
        return inputs_

    def forward_features(
        self, x, inference_params=None, if_random_cls_token_position=False, if_random_token_rank=False
    ):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        x = self.patch_embed(x)
        B, M, _ = x.shape

        if self.if_cls_token:
            if self.use_double_cls_token:
                cls_token_head = self.cls_token_head.expand(B, -1, -1)
                cls_token_tail = self.cls_token_tail.expand(B, -1, -1)
                token_position = [0, M + 1]
                x = torch.cat((cls_token_head, x, cls_token_tail), dim=1)
                M = x.shape[1]
            else:
                if self.use_middle_cls_token:
                    cls_token = self.cls_token.expand(B, -1, -1)
                    token_position = M // 2
                    # add cls token in the middle
                    x = torch.cat((x[:, :token_position, :], cls_token, x[:, token_position:, :]), dim=1)
                elif if_random_cls_token_position:
                    cls_token = self.cls_token.expand(B, -1, -1)
                    token_position = random.randint(0, M)
                    x = torch.cat((x[:, :token_position, :], cls_token, x[:, token_position:, :]), dim=1)
                    print("token_position: ", token_position)
                else:
                    cls_token = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
                    token_position = 0
                    x = torch.cat((cls_token, x), dim=1)
                M = x.shape[1]

        if self.if_abs_pos_embed:
            x = x + self.pos_embed
            x = self.pos_drop(x)

        if if_random_token_rank:
            # general random shuffle index
            shuffle_indices = torch.randperm(M)

            if isinstance(token_position, list):
                print("original value: ", x[0, token_position[0], 0], x[0, token_position[1], 0])
            else:
                print("original value: ", x[0, token_position, 0])
            print("original token_position: ", token_position)

            # execute shuffle
            x = x[:, shuffle_indices, :]

            if isinstance(token_position, list):
                # find new position of cls token after shuffle
                new_token_position = [
                    torch.where(shuffle_indices == token_position[i])[0].item() for i in range(len(token_position))
                ]
                token_position = new_token_position
            else:
                # find new position of cls token after the shuffle
                token_position = torch.where(shuffle_indices == token_position)[0].item()

            if isinstance(token_position, list):
                print("new value: ", x[0, token_position[0], 0], x[0, token_position[1], 0])
            else:
                print("new value: ", x[0, token_position, 0])
            print("new token_position: ", token_position)

        if_flip_img_sequences = False
        if self.flip_img_sequences_ratio > 0 and (self.flip_img_sequences_ratio - random.random()) > 1e-5:
            x = x.flip([1])
            if_flip_img_sequences = True

        # mamba impl
        residual = None
        hidden_states = x
        if not self.if_bidirectional:
            for layer in self.layers:

                if if_flip_img_sequences and self.if_rope:
                    hidden_states = hidden_states.flip([1])
                    if residual is not None:
                        residual = residual.flip([1])

                # rope about
                if self.if_rope:
                    hidden_states = self.rope(hidden_states)
                    if residual is not None and self.if_rope_residual:
                        residual = self.rope(residual)

                if if_flip_img_sequences and self.if_rope:
                    hidden_states = hidden_states.flip([1])
                    if residual is not None:
                        residual = residual.flip([1])

                hidden_states, residual = layer(hidden_states, residual, inference_params=inference_params)
        else:
            # get two layers in a single for-loop
            for i in range(len(self.layers) // 2):
                if self.if_rope:
                    hidden_states = self.rope(hidden_states)
                    if residual is not None and self.if_rope_residual:
                        residual = self.rope(residual)

                hidden_states_f, residual_f = self.layers[i * 2](
                    hidden_states, residual, inference_params=inference_params
                )
                hidden_states_b, residual_b = self.layers[i * 2 + 1](
                    hidden_states.flip([1]),
                    None if residual is None else residual.flip([1]),
                    inference_params=inference_params
                )
                hidden_states = hidden_states_f + hidden_states_b.flip([1])
                residual = residual_f + residual_b.flip([1])

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm = False here since we don't need the residual
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

    def forward(self, x, inference_params=None, if_random_cls_token_position=False, if_random_token_rank=False):
        x = self.forward_features(x, inference_params, if_random_cls_token_position, if_random_token_rank)

        if self.final_pool_type == 'max':
            x = x.max(dim=1)[0]

        if self.if_cls_token:  # remove the class token
            x = x[:, 1:, :]

        # let's get the patches back from the 1d tokens
        x = self.convert_to_expected_dim(x)

        return x  # from here, the tokens can be upsampled easily (N x H x W x C)


def get_vim_encoder(model_type="vim_t", with_cls_token=True):
    if model_type == "vim_t":
        embed_dim = 192
    elif model_type == "vim_s":
        embed_dim = 384
    elif model_type == "vim_b":
        embed_dim = 768
    else:
        raise ValueError("Choose from 'vim_t' / 'vim_s' / 'vim_b'")

    encoder = ViM(
        img_size=1024,
        patch_size=16,
        embed_dim=embed_dim,
        depth=24,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        final_pool_type='all',
        if_abs_pos_embed=True,
        if_rope=False,
        if_rope_residual=False,
        bimamba_type="v2",
        if_cls_token=with_cls_token,
        if_divide_out=True,
        use_middle_cls_token=True,
    )
    encoder.default_cfg = _cfg()
    return encoder


def get_vimunet_model(
    out_channels, model_type="vim_t", with_cls_token=True, device=None, checkpoint=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = get_vim_encoder(model_type, with_cls_token)

    model_state = None
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)

        if checkpoint.endswith(".pth"):  # from Vim
            encoder_state = state["model"]
            encoder.load_state_dict(encoder_state)

        else:  # from torch_em
            model_state = state["model_state"]

    encoder.img_size = encoder.patch_embed.img_size[0]

    # TODO: Update design so that: we have a backbone to fetch encoder and decoder flexibly
    # and is ideally not named as "UNETR" but something as for example "EncoderDecoderNet"
    model = UNETR(
        encoder=encoder,
        out_channels=out_channels,
        resize_input=False,
        use_skip_connection=False,
        final_activation="Sigmoid",
    )

    if model_state is not None:
        model.load_state_dict(model_state)

    model.to(device)

    return model
