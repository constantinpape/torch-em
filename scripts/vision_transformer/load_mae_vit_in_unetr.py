import argparse
from torch_em.model import UNETR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--encoder", default="vit_l")
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()

    UNETR(
        img_size=args.img_size, backbone="mae", encoder=args.encoder, encoder_checkpoint=args.checkpoint
    )
    print("UNETR Model successfully created and encoder initialized from", args.checkpoint)


if __name__ == "__main__":
    main()
