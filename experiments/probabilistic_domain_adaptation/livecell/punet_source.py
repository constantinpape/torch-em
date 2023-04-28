import os
import pandas as pd

import torch

from torch_em.self_training import ProbabilisticUNetTrainer, \
    ProbabilisticUNetLoss, ProbabilisticUNetLossAndMetric

import common

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _train_cell_type(args, cell_type, device=DEVICE):
    train_loader = common.get_supervised_loader(args, "train", cell_type, args.batch_size)
    val_loader = common.get_supervised_loader(args, "val", cell_type, 1)
    name = f"punet_source/{cell_type}"

    model = common.get_punet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=10, verbose=True)
    model.to(device)

    supervised_loss = ProbabilisticUNetLoss()
    supervised_loss_and_metric = ProbabilisticUNetLossAndMetric()

    trainer = ProbabilisticUNetTrainer(
        name=name,
        save_root=args.save_root,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        logger=None,
        device=device,
        lr_scheduler=scheduler,
        optimizer=optimizer,
        mixed_precision=True,
        log_image_interval=100,
        loss=supervised_loss,
        loss_and_metric=supervised_loss_and_metric
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
        res = common.evaluate_source_model(args, ct, "punet_source", get_model=common.get_punet,
                                           prediction_function=common.get_punet_predictions)
        results.append(res)
    results = pd.concat(results)
    print("Evaluation results:")
    print(results)
    result_folder = "./results"
    os.makedirs(result_folder, exist_ok=True)
    results.to_csv(os.path.join(result_folder, "punet_source.csv"), index=False)


def main():
    parser = common.get_parser(default_iterations=100000)
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
