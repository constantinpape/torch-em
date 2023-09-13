import os
import argparse

import torch
import torch_em
from torch_em.model import UNETR
from torch_em.data.datasets import get_livecell_loader


def do_unetr_training(data_path: str, save_root: str, cell_type: list, iterations: int, device, patch_shape=(256, 256)):
    os.makedirs(data_path, exist_ok=True)
    train_loader = get_livecell_loader(
        path=data_path,
        split="train",
        patch_shape=patch_shape,
        batch_size=2,
        cell_types=cell_type,
        download=True,
        boundaries=True
    )

    val_loader = get_livecell_loader(
        path=data_path,
        split="val",
        patch_shape=patch_shape,
        batch_size=1,
        cell_types=cell_type,
        download=True,
        boundaries=True
    )

    n_channels = 2

    model = UNETR(
        encoder="vit_b", out_channels=n_channels,
        encoder_checkpoint_path="/scratch/usr/nimanwai/models/segment-anything/checkpoints/sam_vit_b_01ec64.pth")
    model.to(device)

    trainer = torch_em.default_segmentation_trainer(
        name=f"unetr-source-livecell-{cell_type[0]}",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=1e-5,
        log_image_interval=10,
        save_root=save_root,
        compile_model=False
    )

    trainer.fit(iterations)


def main(args):
    print(torch.cuda.get_device_name() if torch.cuda.is_available() else "GPU not available, hence running on CPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.train:
        print("Training a 2D UNETR on LiveCell dataset")
        do_unetr_training(
            data_path=args.inputs,
            save_root=args.save_root,
            cell_type=args.cell_type,
            iterations=args.iterations,
            device=device
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action='store_true', help="Enables UNETR training on LiveCELL dataset")
    parser.add_argument("-c", "--cell_type", nargs='+', default=["A172"],
                        help="Choice of cell-type for doing the training")
    parser.add_argument("-i", "--inputs", type=str, default="./livecell/",
                        help="Path where the dataset already exists/will be downloaded by the dataloader")
    parser.add_argument("-s", "--save_root", type=str, default=None,
                        help="Path where checkpoints and logs will be saved")
    parser.add_argument("--iterations", type=int, default=100000, help="No. of iterations to run the training for")
    args = parser.parse_args()
    main(args)
