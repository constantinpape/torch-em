import torch

from torch_em.model.unetr import get_unetr_model


def get_input(ndim, device):
    shape = (1, 1, 256, 256) if ndim == 2 else (1, 1, 4, 256, 256)
    return torch.rand(shape, device=device)


def load_torchvision(ndim, model_type, pretrained):
    """Load a torchvision ViT encoder in UNETR."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # The factory disables the skip connections for the encoders with patch size 32 and 14.
    model = get_unetr_model(
        ndim=ndim, backbone="torchvision", model_type=model_type, pretrained=pretrained,
        out_channels=1, final_activation="Sigmoid",
    ).to(device)

    x = get_input(ndim, device)
    with torch.no_grad():
        y = model(x)
    print(x.shape, y.shape)
    print(f"UNETR with the torchvision {model_type} encoder was created with pretrained={pretrained}.")


def main():
    # The torchvision models are: vit_b_16, vit_b_32, vit_l_16, vit_l_32, and vit_h_14.
    load_torchvision(ndim=2, model_type="vit_b_16", pretrained=True)
    load_torchvision(ndim=3, model_type="vit_b_16", pretrained=True)

    load_torchvision(ndim=2, model_type="vit_b_32", pretrained=False)
    load_torchvision(ndim=3, model_type="vit_b_32", pretrained=False)


if __name__ == "__main__":
    main()
