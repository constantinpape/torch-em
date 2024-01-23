import os
import argparse

import torch_em
from torch_em.data.datasets import get_livecell_loader

from vimunet import get_vimunet_model


ROOT = "/scratch/usr/nimanwai"


def get_loaders(path):
    patch_shape = (520, 704)

    train_loader = get_livecell_loader(
        path=path, split="train", patch_shape=patch_shape, batch_size=2, binary=True, cell_types=["A172"],
    )

    val_loader = get_livecell_loader(
        path=path, split="val", patch_shape=patch_shape, batch_size=1, binary=True, cell_types=["A172"],
    )

    return train_loader, val_loader


def run_livecell_training(args):
    # the dataloaders for livecell dataset
    train_loader, val_loader = get_loaders(path=args.input)

    # the vision-mamba + decoder (UNet-based) model
    model = get_vimunet_model(checkpoint=args.checkpoint)

    # trainer for the segmentation task
    trainer = torch_em.default_segmentation_trainer(
        name="livecell-vimunet",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=1e-3,
        save_root=args.save_root,
        compile_model=False
    )
    trainer.fit(iterations=int(args.iterations))


def main(args):
    run_livecell_training(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default=os.path.join(ROOT, "data", "livecell"))
    parser.add_argument("--iterations", type=int, default=1e4)
    parser.add_argument("-s", "--save_root", type=str, default=os.path.join(ROOT, "experiments", "vision-mamba"))
    parser.add_argument(
        "--checkpoint", type=str, default=os.path.join(ROOT, "models", "Vim-tiny", "vim_tiny_73p1.pth")
    )
    args = parser.parse_args()
    main(args)
