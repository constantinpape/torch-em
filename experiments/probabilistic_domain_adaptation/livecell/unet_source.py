import os

import torch_em
import pandas as pd

import common


def _train_cell_type(args, cell_type):
    model = common.get_unet()
    train_loader = common.get_supervised_loader(args, "train", cell_type)
    val_loader = common.get_supervised_loader(args, "val", cell_type)
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

    loader = common.get_supervised_loader(args)
    check_loader(loader, n_images)


def run_evaluation(args):
    results = []
    for ct in args.cell_types:
        res = common.evaluate_source_model(args, ct, "unet_source")
        results.append(res)
    results = pd.concat(results)
    print("Evaluation results:")
    print(results)
    result_folder = "./results"
    os.makedirs(result_folder, exist_ok=True)
    results.to_csv(os.path.join(result_folder, "unet_source.csv"), index=False)


def main():
    parser = common.get_parser(default_iterations=50000)
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
