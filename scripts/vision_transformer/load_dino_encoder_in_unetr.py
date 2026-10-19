import os

import torch

from torch_em.model.unetr import get_unetr_model


def get_input(ndim, device):
    shape = (1, 1, 256, 256) if ndim == 2 else (1, 1, 4, 256, 256)
    return torch.rand(shape, device=device)


def load_dino(ndim, backbone, model_type, checkpoint):
    """Load a DINO encoder in UNETR with pretrained weights."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = get_unetr_model(
        ndim=ndim, backbone=backbone, model_type=model_type, encoder_checkpoint=checkpoint,
        out_channels=1, final_activation="Sigmoid",
    ).to(device)

    x = get_input(ndim, device)
    with torch.no_grad():
        y = model(x)
    print(x.shape, y.shape)
    print(f"UNETR with the {backbone} {model_type} encoder loaded the weights from", checkpoint)


def main():
    dinov3_checkpoint = os.path.expanduser("~/models/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")
    load_dino(ndim=2, backbone="dinov3", model_type="vit_b", checkpoint=dinov3_checkpoint)
    load_dino(ndim=3, backbone="dinov3", model_type="vit_b", checkpoint=dinov3_checkpoint)

    # The other DINOv2 checkpoints and their model types:
    # ./dinov2_vitb14_pretrain.pth with vit_b, ./dinov2_vitb14_reg4_pretrain.pth with vit_b_reg4
    # ./dinov2_vitl14_pretrain.pth with vit_l, ./dinov2_vitl14_reg4_pretrain.pth with vit_l_reg4
    # ./dinov2_vitg14_pretrain.pth with vit_g, ./dinov2_vitg14_reg4_pretrain.pth with vit_g_reg4
    load_dino(ndim=2, backbone="dinov2", model_type="vit_s", checkpoint="./dinov2_vits14_pretrain.pth")
    load_dino(ndim=2, backbone="dinov2", model_type="vit_s_reg4", checkpoint="./dinov2_vits14_reg4_pretrain.pth")
    load_dino(ndim=3, backbone="dinov2", model_type="vit_s_reg4", checkpoint="./dinov2_vits14_reg4_pretrain.pth")


if __name__ == "__main__":
    main()
