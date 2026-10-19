import torch

from torch_em.model.unetr import get_unetr_model


def get_input(ndim, device):
    # CellposeSAM uses vit_l with patch_size=8 and img_size=256 by default.
    shape = (1, 1, 256, 256) if ndim == 2 else (1, 1, 4, 256, 256)
    return torch.rand(shape, device=device)


def load_cellpose_sam(ndim):
    """Load the CellposeSAM encoder in UNETR without pretrained weights."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = get_unetr_model(
        ndim=ndim, backbone="cellpose_sam", model_type="vit_l", out_channels=3, final_activation="Sigmoid"
    ).to(device)

    x = get_input(ndim, device)
    with torch.no_grad():
        y = model(x)
    print(x.shape, y.shape)
    print("UNETR with the CellposeSAM encoder was created.")


def load_cellpose_sam_pretrained(ndim):
    """Load the CellposeSAM encoder in UNETR with pretrained weights."""
    from cellpose.models import MODEL_DIR

    checkpoint = MODEL_DIR / "cpsam"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = get_unetr_model(
        ndim=ndim, backbone="cellpose_sam", model_type="vit_l", encoder_checkpoint=checkpoint,
        out_channels=3, final_activation="Sigmoid",
    ).to(device)

    x = get_input(ndim, device)
    with torch.no_grad():
        y = model(x)
    print(x.shape, y.shape)
    print("UNETR with the CellposeSAM encoder loaded the weights from", checkpoint)


def main():
    load_cellpose_sam(ndim=2)
    load_cellpose_sam_pretrained(ndim=2)

    load_cellpose_sam(ndim=3)
    load_cellpose_sam_pretrained(ndim=3)


if __name__ == "__main__":
    main()
