import torch

from torch_em.model import UNETR


def main():
    # checkpoint_path = "./dinov2_vits14_pretrain.pth"
    # model_type = "vit_s"

    checkpoint_path = "./dinov2_vits14_reg4_pretrain.pth"
    model_type = "vit_s_reg4"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = UNETR(
        backbone="dinov2",
        encoder=model_type,
        encoder_checkpoint=checkpoint_path,
        out_channels=1,
        use_dino_stats=True,
        final_activation="Sigmoid",
        use_skip_connection=False,
        resize_input=True,
        img_size=518,
    )
    model.to(device)

    x = torch.ones((1, 3, 256, 256)).to(device)
    y = model(x)

    breakpoint()

    print(x.shape, y.shape)


if __name__ == "__main__":
    main()
