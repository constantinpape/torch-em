import argparse

import torch_em
from torch_em.model import UNet2d
from torch_em.data.datasets import get_livecell_loader

CELL_TYPES = ["A172", "BT474", "BV2", "Huh7", "MCF7", "SHSY5Y", "SkBr3", "SKOV3"]


def _get_loader(args, split, cell_type):
    patch_shape = (512, 512)
    loader = get_livecell_loader(
        args.input, patch_shape, split,
        download=True, binary=True, batch_size=args.batch_size,
        cell_types=[cell_type]
    )
    return loader


def _train_cell_type(args, cell_type):
    model = UNet2d(in_channels=1, out_channels=1, initial_features=64, final_activation="Sigmoid")
    train_loader = _get_loader(args, "train", cell_type)
    val_loader = _get_loader(args, "train", cell_type)
    name = f"unet_source/{cell_type}"
    trainer = torch_em.default_segmentation_trainer(
        name=name,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=1e-4,
        mixed_precision=True,
        log_image_interval=100,
    )
    trainer.fit(iterations=args.n_iterations)


def run_training(args):
    for cell_type in args.cell_types:
        print("Start training for cell type:", cell_type)
        _train_cell_type(args, cell_type)


def check_loader(args, n_images=5):
    from torch_em.util.debug import check_loader

    cell_types = args.cell_types
    print("The cell types", cell_types, "were selected.")
    print("Checking the loader for the first cell type", cell_types[0])

    loader = _get_loader(args)
    check_loader(loader, n_images)


# TODO
def run_evaluation(args):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-p", "--phase", required=True)
    parser.add_argument("-b", "--batch_size", default=8, type=int)
    parser.add_argument("-n", "--n_iterations", default=int(1e5), type=int)
    parser.add_argument("--cell_types", nargs="+", default=CELL_TYPES)
    args = parser.parse_args()

    phase = args.phase
    if phase in ("c", "check"):
        check_loader(args)
    elif phase in ("t", "train"):
        run_training(args)
    elif phase in ("e", "evaluate"):
        run_evaluation(args)
    else:
        raise ValueError(f"Got phase={phase}, expect one of check, train, evaluate.")


if __name__ == "__main__":
    main()
