import torch

from torch_em.model import UNETR

from micro_sam.util import get_sam_model


def main():
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


if __name__ == "__main__":
    main()
