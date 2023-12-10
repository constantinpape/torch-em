from torch_em.model import UNETR


def main():
    checkpoint = "/home/nimanwai/mae_models/imagenet.pth"
    unetr_model = UNETR(img_size=224, backbone="mae", encoder="vit_l", encoder_checkpoint=checkpoint)
    print(unetr_model)


if __name__ == "__main__":
    main()
