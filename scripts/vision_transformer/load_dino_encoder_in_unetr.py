import argparse

import imageio.v3 as imageio

import torch

from torch_em.model.unetr import get_unetr_model
from torch_em.transform.raw import get_raw_transform_for_pretrained_vit


def load_input(input_path, backbone, ndim, device):
    raw_transform, _ = get_raw_transform_for_pretrained_vit(backbone)
    raw = raw_transform(imageio.imread(input_path))
    if raw.ndim == 2 and ndim == 3:
        raw = raw[None]
    if raw.ndim != ndim:
        raise ValueError(f"The input shape {raw.shape} does not match ndim={ndim}. Pass a {ndim}D grayscale input.")
    return torch.as_tensor(raw, dtype=torch.float32, device=device)[None, None]


def check_unetr(checkpoint, input_path, backbone, model_type, ndim, initial_features):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_unetr_model(
        ndim=ndim, backbone=backbone, model_type=model_type,
        encoder_checkpoint=checkpoint, initial_features=initial_features
    ).to(device).eval()
    x = load_input(input_path, backbone, ndim, device)
    with torch.inference_mode():
        y = model(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    print(f"Input: {tuple(x.shape)}, output: {tuple(y.shape)}")


def main():
    # DINOv2 checkpoints and their matching --model-type values:
    # --checkpoint ./dinov2_vits14_pretrain.pth --model-type vit_s
    # --checkpoint ./dinov2_vits14_reg4_pretrain.pth --model-type vit_s_reg4
    # --checkpoint ./dinov2_vitb14_pretrain.pth --model-type vit_b
    # --checkpoint ./dinov2_vitb14_reg4_pretrain.pth --model-type vit_b_reg4
    # --checkpoint ./dinov2_vitl14_pretrain.pth --model-type vit_l
    # --checkpoint ./dinov2_vitl14_reg4_pretrain.pth --model-type vit_l_reg4
    # --checkpoint ./dinov2_vitg14_pretrain.pth --model-type vit_g
    # --checkpoint ./dinov2_vitg14_reg4_pretrain.pth --model-type vit_g_reg4
    parser = argparse.ArgumentParser(description="Load a pretrained DINO encoder in a 2D or 3D UNETR.")
    parser.add_argument("--checkpoint", required=True, help="The path to the encoder checkpoint.")
    parser.add_argument("--input", required=True, help="The path to a grayscale image or volume.")
    parser.add_argument("--backbone", choices=("dinov2", "dinov3"), default="dinov2")
    parser.add_argument("--model-type", default="vit_s_reg4")
    parser.add_argument("--ndim", type=int, choices=(2, 3), default=2)
    parser.add_argument("--initial-features", type=int, default=64)
    args = parser.parse_args()
    check_unetr(args.checkpoint, args.input, args.backbone, args.model_type, args.ndim, args.initial_features)


if __name__ == "__main__":
    main()
