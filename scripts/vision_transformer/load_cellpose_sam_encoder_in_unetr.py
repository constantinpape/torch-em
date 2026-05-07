import torch

from torch_em.model.unetr import UNETR2D, UNETR3D


def load_cellpose_sam(ndim):
    """Load CellposeSAM image encoder in UNETR (without pretrained weights)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # CellposeSAM uses vit_l with patch_size=8 and img_size=256 by default.

    if ndim == 2:
        x = torch.ones((1, 1, 256, 256)).to(device)
    else:
        x = torch.ones((1, 1, 4, 256, 256)).to(device)

    if ndim == 2:
        model = UNETR2D(
            backbone="cellpose_sam",
            encoder="vit_l",
            img_size=256,
            out_channels=3,
            use_sam_stats=True,
            final_activation="Sigmoid",
            use_skip_connection=False,
        )
    else:
        model = UNETR3D(
            backbone="cellpose_sam",
            encoder="vit_l",
            img_size=256,
            out_channels=3,
            use_sam_stats=True,
            final_activation="Sigmoid",
            use_skip_connection=False,
            embed_dim=1024,
        )

    model.to(device)

    y = model(x)
    print(x.shape, y.shape)
    print("UNETR Model with CellposeSAM encoder successfully created.")


def load_cellpose_sam_pretrained(ndim):
    """Load CellposeSAM image encoder in UNETR with pretrained weights."""
    from cellpose.models import model_path

    checkpoint = model_path("cpsam")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if ndim == 2:
        x = torch.ones((1, 1, 256, 256)).to(device)
    else:
        x = torch.ones((1, 1, 4, 256, 256)).to(device)

    if ndim == 2:
        model = UNETR2D(
            backbone="cellpose_sam",
            encoder="vit_l",
            img_size=256,
            encoder_checkpoint=checkpoint,
            out_channels=3,
            use_sam_stats=True,
            final_activation="Sigmoid",
            use_skip_connection=False,
        )
    else:
        model = UNETR3D(
            backbone="cellpose_sam",
            encoder="vit_l",
            img_size=256,
            encoder_checkpoint=checkpoint,
            out_channels=3,
            use_sam_stats=True,
            final_activation="Sigmoid",
            use_skip_connection=False,
            embed_dim=1024,
        )

    model.to(device)

    y = model(x)
    print(x.shape, y.shape)
    print("UNETR Model with pretrained CellposeSAM encoder successfully created from", checkpoint)


def main():
    load_cellpose_sam(ndim=2)
    load_cellpose_sam_pretrained(ndim=2)

    load_cellpose_sam(ndim=3)
    load_cellpose_sam_pretrained(ndim=3)


if __name__ == "__main__":
    main()
