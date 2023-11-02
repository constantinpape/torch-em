import argparse
from typing import Tuple

from torch_em.data.datasets import get_livecell_loader
from torch_em.loss import DiceLoss, LossWrapper, ApplyAndRemoveMask


OFFSETS = [
    [-1, 0], [0, -1],
    [-3, 0], [0, -3],
    [-9, 0], [0, -9],
    [-27, 0], [0, -27]
]


CELL_TYPES = ["A172", "BT474", "BV2", "Huh7", "MCF7", "SHSY5Y", "SkBr3", "SKOV3"]


#
# LIVECELL DATALOADERS
#

def get_my_livecell_loaders(
        input_path: str,
        patch_shape: Tuple[int, int],
        cell_types: str,
        with_affinities: bool = False
):
    """Returns the LIVECell training and validation dataloaders
    """
    if with_affinities:
        # this returns dataloaders with affinity channels and foreground-background channels
        n_out = len(OFFSETS) + 1
        train_loader = get_livecell_loader(
            path=input_path, split="train", patch_shape=patch_shape, batch_size=2,
            cell_types=[cell_types], download=True, offsets=OFFSETS, num_workers=16
        )
        val_loader = get_livecell_loader(
            path=input_path, split="val", patch_shape=patch_shape, batch_size=1,
            cell_types=[cell_types], download=True, offsets=OFFSETS, num_workers=16
        )

    else:
        # this returns dataloaders with foreground and boundary channels
        n_out = 2
        train_loader = get_livecell_loader(
            path=input_path, split="train", patch_shape=patch_shape, batch_size=2,
            cell_types=[cell_types], download=True, boundaries=True, num_workers=16
        )
        val_loader = get_livecell_loader(
            path=input_path, split="val", patch_shape=patch_shape, batch_size=1,
            cell_types=[cell_types], download=True, boundaries=True, num_workers=16
        )

    return train_loader, val_loader, n_out


#
# UNETR MODEL(S) FROM MONAI AND torch_em
#


def get_unetr_model(
        model_name: str,
        source_choice: str,
        patch_shape: Tuple[int, int],
        sam_initialization: bool,
        output_channels: int
):
    """Returns the expected UNETR model
    """
    if source_choice == "torch-em":
        # this returns the unetr model whihc uses the vision transformer from segment anything
        from torch_em import model as torch_em_models
        model = torch_em_models.UNETR(
            encoder=model_name, out_channels=output_channels,
            encoder_checkpoint_path="/scratch/usr/nimanwai/models/segment-anything/checkpoints/sam_vit_b_01ec64.pth" if sam_initialization else None
        )

    elif source_choice == "monai":
        # this returns the unetr model from monai
        from monai.networks import nets as monai_models
        model = monai_models.unetr.UNETR(
            in_channels=1,
            out_channels=output_channels,
            img_size=patch_shape,
            spatial_dims=2
        )
        model.out_channels = 2  # type: ignore

    else:
        raise ValueError(f"The available UNETR models are either from \"torch-em\" or \"monai\", choose from them instead of - {source_choice}")

    return model


#
# miscellanous utilities
#


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train", action='store_true', help="Enables UNETR training on LiveCell dataset"
    )

    parser.add_argument(
        "--predict", action='store_true', help="Enables UNETR prediction on LiveCell dataset"
    )

    parser.add_argument(
        "--evaluate", action='store_true', help="Enables UNETR evaluation on LiveCell dataset"
    )

    parser.add_argument(
        "--source_choice", type=str, default="torch-em",
        help="The source where the model comes from, i.e. either torch-em / monai"
    )

    parser.add_argument(
        "-m", "--model_name", type=str, default="vit_b", help="Name of the ViT to use as the encoder in UNETR"
    )

    parser.add_argument(
        "--do_sam_ini", action='store_true', help="Enables initializing UNETR with SAM's ViT weights"
    )

    parser.add_argument(
        "-c", "--cell_type", type=str, default=None, help="Choice of cell-type for doing the training"
    )

    parser.add_argument(
        "-i", "--input", type=str, default="/scratch/usr/nimanwai/data/livecell",
        help="Path where the dataset already exists/will be downloaded by the dataloader"
    )

    parser.add_argument(
        "-s", "--save_root", type=str, default="/scratch/usr/nimanwai/models/unetr/",
        help="Path where checkpoints and logs will be saved"
    )

    parser.add_argument(
        "--save_dir", type=str, default="/scratch/usr/nimanwai/predictions/unetr",
        help="Path to save predictions from UNETR model"
    )

    parser.add_argument(
        "--with_affinities", action="store_true",
        help="Trains the UNETR model with affinities"
    )

    parser.add_argument("--iterations", type=int, default=100000)
    return parser


def get_loss_function(with_affinities=True):
    if with_affinities:
        loss = LossWrapper(
            loss=DiceLoss(),
            transform=ApplyAndRemoveMask(masking_method="multiply")
        )
    else:
        loss = DiceLoss()

    return loss
