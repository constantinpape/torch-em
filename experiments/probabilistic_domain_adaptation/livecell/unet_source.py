import torch_em
from torch_em.model import UNet2d
from common import get_parser, get_supervised_loader


def _train_cell_type(args, cell_type):
    model = UNet2d(in_channels=1, out_channels=1, initial_features=64, final_activation="Sigmoid")
    train_loader = get_supervised_loader(args, "train", cell_type)
    val_loader = get_supervised_loader(args, "val", cell_type)
    name = f"unet_source/{cell_type}"
    trainer = torch_em.default_segmentation_trainer(
        name=name,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=1e-4,
        mixed_precision=True,
        log_image_interval=100,
        save_root=args.save_root,
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

    loader = get_supervised_loader(args)
    check_loader(loader, n_images)


# TODO
def run_evaluation(args):
    pass


def main():
    parser = get_parser(default_iterations=50000)
    args = parser.parse_args()
    if args.phase in ("c", "check"):
        check_loader(args)
    elif args.phase in ("t", "train"):
        run_training(args)
    elif args.phase in ("e", "evaluate"):
        run_evaluation(args)
    else:
        raise ValueError(f"Got phase={args.phase}, expect one of check, train, evaluate.")


if __name__ == "__main__":
    main()
