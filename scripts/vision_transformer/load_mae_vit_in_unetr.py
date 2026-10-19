import torch

from torch_em.model.unetr import get_unetr_model


def get_input(ndim, device):
    shape = (1, 1, 256, 256) if ndim == 2 else (1, 1, 4, 256, 256)
    return torch.rand(shape, device=device)


def load_mae(ndim, backbone, model_type, checkpoint=None):
    """Load a MAE or ScaleMAE encoder in UNETR."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = get_unetr_model(
        ndim=ndim, backbone=backbone, model_type=model_type, encoder_checkpoint=checkpoint,
        out_channels=1, final_activation="Sigmoid",
    ).to(device)

    x = get_input(ndim, device)
    with torch.no_grad():
        y = model(x)
    print(x.shape, y.shape)
    print(f"UNETR with the {backbone} {model_type} encoder was created with checkpoint={checkpoint}.")


def main():
    # Pass the path to a pretrained checkpoint as `checkpoint` to load the encoder weights.
    load_mae(ndim=2, backbone="mae", model_type="vit_b")
    load_mae(ndim=3, backbone="mae", model_type="vit_b")

    load_mae(ndim=2, backbone="scalemae", model_type="vit_b")
    load_mae(ndim=3, backbone="scalemae", model_type="vit_b")


if __name__ == "__main__":
    main()
