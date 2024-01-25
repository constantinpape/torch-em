import os
import argparse
from glob import glob

import imageio.v3 as imageio

import torch

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

    if args.pretrained:
        checkpoint = "/scratch/usr/nimanwai/models/Vim-tiny/vim_tiny_73p1.pth"
    else:
        checkpoint = None

    # the vision-mamba + decoder (UNet-based) model
    model = get_vimunet_model(checkpoint=checkpoint)

    # saving the model checkpoints
    save_root = os.path.join(
        args.save_root,
        "pretrained" if args.pretrained else "scratch"
    )

    # trainer for the segmentation task
    trainer = torch_em.default_segmentation_trainer(
        name="livecell-vimunet",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=1e-4,
        save_root=save_root,
        compile_model=False
    )
    trainer.fit(iterations=int(args.iterations))


def run_livecell_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # the vision-mamba + decoder (UNet-based) model
    model = get_vimunet_model(checkpoint=args.checkpoint)

    for image_path in glob(os.path.join(ROOT, "data", "livecell", "images", "livecell_test_images", "*")):
        image = imageio.imread(image_path)
        tensor_image = torch.from_numpy(image)[None, None].to(device)

        predictions = model(tensor_image)
        predictions = predictions.squeeze().detach().cpu().numpy()


def main(args):
    if args.train:
        run_livecell_training(args)

    if args.predict:
        assert args.checkpoint is not None, "Provide the checkpoint path to the trained model."
        run_livecell_inference(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default=os.path.join(ROOT, "data", "livecell"))
    parser.add_argument("--iterations", type=int, default=1e4)
    parser.add_argument("-s", "--save_root", type=str, default=os.path.join(ROOT, "experiments", "vision-mamba"))
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("-c", "--checkpoint", default=None, type=str)
    args = parser.parse_args()
    main(args)
