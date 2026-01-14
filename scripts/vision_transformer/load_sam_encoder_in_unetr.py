import torch

from torch_em.model import UNETR

from micro_sam.util import get_sam_model


def load_sam1():
    checkpoint = "/scratch/usr/nimanwai/models/segment-anything/checkpoints/sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    predictor = get_sam_model(
        model_type=model_type,
        checkpoint_path=checkpoint
    )

    model = UNETR(
        backbone="sam",
        encoder=predictor.model.image_encoder,
        out_channels=3,
        use_sam_stats=True,
        final_activation="Sigmoid",
        use_skip_connection=False
    )
    model.to(device)

    x = torch.ones((1, 1, 512, 512)).to(device)
    y = model(x)
    print(x.shape, y.shape)

    print("UNETR Model successfully created and encoder initialized from", checkpoint)


def load_sam2():
    from micro_sam2.util import _get_checkpoint

    model_type = "hvit_t"
    checkpoint = _get_checkpoint(model_type=model_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNETR(
        backbone="sam2",
        encoder=model_type,
        out_channels=3,
        use_sam_stats=True,
        final_activation="Sigmoid",
        use_skip_connections=False,
        embed_dim=256,
    )
    model.to(device)

    x = torch.ones((1, 1, 512, 512)).to(device)
    y = model(x)
    print(x.shape, y.shape)

    checkpoint = None  # HACK
    print("UNETR Model successfully created and encoder initialized from", checkpoint)


def load_sam3():
    from micro_sam3.util import _get_checkpoint

    model_type = "vit_pe"
    checkpoint = _get_checkpoint()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = UNETR(
        backbone="sam3",
        encoder=model_type,
        encoder_checkpoint=checkpoint,
        out_channels=3,
        use_sam_stats=True,
        final_activation="Sigmoid",
        use_skip_connection=False,
    )
    model.to(device)

    x = torch.ones((1, 1, 512, 512)).to(device)
    y = model(x)
    print(x.shape, y.shape)

    checkpoint = None  # HACK
    print("UNETR Model successfully created and encoder initialized from", checkpoint)


def main():
    # load_sam1()
    # load_sam2()
    load_sam3()


if __name__ == "__main__":
    main()
