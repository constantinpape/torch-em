import argparse

import torch
from torch_em.model import UNETR


def main():
    parser = argparse.ArgumentParser(description="Load a torchvision ViT encoder in UNETR.")
    parser.add_argument(
        "--encoder", default="vit_b_16",
        help=(
            "Torchvision ViT variant: vit_b_16, vit_b_32, vit_l_16, vit_l_32, vit_h_14. "
            "vit_b_32, vit_l_32, and vit_h_14 require use_skip_connection=False (set automatically)."
        ),
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help=(
            "Optional path to a fine-tuned torchvision ViT checkpoint (.pth). "
            "If not provided, ImageNet-pretrained weights are loaded from torchvision."
        ),
    )
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--out_channels", type=int, default=1)
    parser.add_argument("--no_pretrained", action="store_true", help="Skip loading pretrained weights.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # patch_size=32 and patch_size=14 models need skip connections disabled so that
    # all tensors in the decoder chain from the same patch grid and sizes align.
    # postprocess_masks then resizes the output back to the input resolution.
    use_skip_connection = args.encoder.endswith("_16")

    model = UNETR(
        backbone="torchvision",
        encoder=args.encoder,
        img_size=args.img_size,
        out_channels=args.out_channels,
        use_imagenet_stats=True,
        final_activation="Sigmoid",
        resize_input=True,
        pretrained=not args.no_pretrained,
        encoder_checkpoint=args.checkpoint,
        use_skip_connection=use_skip_connection,
    )
    model.to(device)

    x = torch.rand(1, 3, args.img_size, args.img_size).to(device)
    y = model(x)

    print(f"Encoder: {args.encoder}")
    print(f"Input: {tuple(x.shape)}")
    print(f"Output: {tuple(y.shape)}")


if __name__ == "__main__":
    main()
