import argparse

import imageio.v3 as imageio

import torch

from torch_em.model import vit
from torch_em.model.util import VIT_CONFIGS
from torch_em.model.unetr import get_unetr_model
from torch_em.transform.raw import get_raw_transform_for_pretrained_vit


def list_backends():
    for backbone, config in VIT_CONFIGS.items():
        status = "available" if getattr(vit, f"_{config.dependency}_import_success") else "unavailable"
        print(f"{backbone}: {status}")


def load_input(input_path, backbone, ndim, device):
    raw_transform, _ = get_raw_transform_for_pretrained_vit(backbone)
    raw = raw_transform(imageio.imread(input_path))
    if raw.ndim == 2 and ndim == 3:
        raw = raw[None]
    if raw.ndim != ndim:
        raise ValueError(f"The input shape {raw.shape} does not match ndim={ndim}. Pass a {ndim}D grayscale input.")
    return torch.as_tensor(raw, dtype=torch.float32, device=device)[None, None]


def check_model(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kwargs = {"initial_features": args.initial_features}
    if args.img_size is not None:
        kwargs["img_size"] = args.img_size
    if args.backbone == "torchvision":
        kwargs["pretrained"] = not args.no_pretrained and args.checkpoint is None
    model = get_unetr_model(
        ndim=args.ndim, backbone=args.backbone, model_type=args.model_type,
        encoder_checkpoint=args.checkpoint, **kwargs
    ).to(device).eval()
    x = load_input(args.input, args.backbone, args.ndim, device)
    with torch.inference_mode():
        y = model(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    print(f"Backbone: {args.backbone}, encoder: {args.model_type}, dimensions: {args.ndim}")
    print(f"Encoder width: {model.embed_dim}, input size: {model.encoder.img_size}")
    print(f"Input: {tuple(x.shape)}, output: {tuple(y.shape)}")


def main():
    parser = argparse.ArgumentParser(description="Check UNETR with a real image and an optional encoder checkpoint.")
    parser.add_argument("--list-backends", action="store_true")
    parser.add_argument("--backbone", choices=tuple(VIT_CONFIGS), default="sam")
    parser.add_argument("--model-type")
    parser.add_argument("--checkpoint")
    parser.add_argument("--input", help="The path to a grayscale image or volume.")
    parser.add_argument("--ndim", type=int, choices=(2, 3), default=2)
    parser.add_argument("--img-size", type=int)
    parser.add_argument("--initial-features", type=int, default=4)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()
    if args.list_backends:
        list_backends()
        return
    if args.input is None or args.model_type is None:
        parser.error("--input and --model-type are required to check a model.")
    check_model(args)


if __name__ == "__main__":
    main()
