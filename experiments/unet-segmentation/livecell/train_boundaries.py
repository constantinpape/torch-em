import torch
import torch_em
from torch_em.model import UNet2d
from torch_em.data.datasets import get_livecell_loader


def train_boundaries(args):
    # Get the UNet model.
    n_out = 2
    model = UNet2d(
        in_channels=1,
        out_channels=n_out,
        initial_features=64,
        final_activation="Sigmoid",
        norm="BatchRenorm",  # HACK: testing a new custom normalization.
    )

    # Get the dataloaders.
    patch_shape = (512, 512)
    train_loader = get_livecell_loader(
        path=args.input,
        patch_shape=patch_shape,
        split="train",
        download=True,
        boundaries=True,
        batch_size=args.batch_size,
        cell_types=None if args.cell_type is None else [args.cell_type],
    )
    val_loader = get_livecell_loader(
        path=args.input,
        patch_shape=patch_shape,
        split="val",
        boundaries=True,
        batch_size=args.batch_size,
        cell_types=None if args.cell_type is None else [args.cell_type],
    )

    # Get the loss function and other stuff for training.
    loss = torch_em.loss.DiceLoss()

    cell_type = args.cell_type
    name = "livecell-boundary-model"
    if cell_type is not None:
        name = f"{name}-{cell_type}"

    # Get the trainer.
    trainer = torch_em.default_segmentation_trainer(
        name=name,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss=loss,
        metric=loss,
        learning_rate=1e-4,
        device=torch.device("cuda"),
        mixed_precision=False,
        log_image_interval=50,
        compile_model=False,
    )
    trainer.fit(iterations=args.n_iterations)


def check_loader(args, train=True, val=True, n_images=5):
    from torch_em.util.debug import check_loader
    patch_shape = (512, 512)
    if train:
        print("Check train loader")
        loader = get_livecell_loader(
            path=args.input,
            patch_shape=patch_shape,
            split="train",
            download=True,
            boundaries=True,
            batch_size=1,
            cell_types=None if args.cell_type is None else [args.cell_type],
        )
        check_loader(loader, n_images)
    if val:
        print("Check val loader")
        loader = get_livecell_loader(
            path=args.input,
            patch_shape=patch_shape,
            split="val",
            download=True,
            boundaries=True,
            batch_size=1,
            cell_types=None if args.cell_type is None else [args.cell_type],
        )
        check_loader(loader, n_images)


if __name__ == "__main__":
    parser = torch_em.util.parser_helper(default_batch_size=8)
    parser.add_argument("--cell_type", default=None)
    args = parser.parse_args()

    if args.check:
        check_loader(args)
    else:
        train_boundaries(args)
