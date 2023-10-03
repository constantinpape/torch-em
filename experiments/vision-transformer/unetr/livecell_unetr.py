import os
import argparse

import torch
import torch_em
from torch_em.model import UNETR
from torch_em.data.datasets import get_livecell_loader


def do_unetr_training(args, device, model_choice="vit_b", patch_shape=(512, 512)):
    os.makedirs(args.input, exist_ok=True)
    train_loader = get_livecell_loader(
        path=args.input,
        split="train",
        patch_shape=patch_shape,
        batch_size=2,
        cell_types=[args.cell_type],
        download=True,
        boundaries=True,
        num_workers=8
    )

    val_loader = get_livecell_loader(
        path=args.input,
        split="val",
        patch_shape=patch_shape,
        batch_size=1,
        cell_types=[args.cell_type],
        download=True,
        boundaries=True,
        num_workers=8
    )

    n_channels = 2

    model = UNETR(
        encoder=model_choice, out_channels=n_channels,
        encoder_checkpoint_path="/scratch/usr/nimanwai/models/segment-anything/checkpoints/sam_vit_b_01ec64.pth" if args.do_sam_ini else None)
    model.to(device)

    _name = "livecell-unetr" if args.cell_type is None else f"livecell-{args.cell_type}-unetr"

    _save_root = os.path.join(
        args.save_root,
        f"sam-{model_choice}" if args.do_sam_ini else "scratch"
    ) if args.save_root is not None else args.save_root
    _save_root = os.path.join(_save_root, args.cell_type) if args.save_root is not None else _save_root

    trainer = torch_em.default_segmentation_trainer(
        name=_name,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=1e-4,
        device=device,
        mixed_precision=True,
        log_image_interval=50,
        compile_model=False,
        save_root=_save_root,
    )

    trainer.fit(args.iterations)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.train:
        print("Training a 2D UNETR on LiveCell dataset")
        do_unetr_training(args, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action='store_true', help="Enables UNETR training on LiveCell dataset")
    parser.add_argument("--do_sam_ini", action='store_true',
                        help="Enables initializing UNETR with SAM's ViT weights")
    parser.add_argument("-c", "--cell_type", type=str, default=None,
                        help="Choice of cell-type for doing the training")
    parser.add_argument("-i", "--input", type=str, default="./livecell/",
                        help="Path where the dataset already exists/will be downloaded by the dataloader")
    parser.add_argument("-s", "--save_root", type=str, default=None,
                        help="Path where checkpoints and logs will be saved")
    parser.add_argument("--iterations", type=int, default=100000)
    args = parser.parse_args()
    main(args)
