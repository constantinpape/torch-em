import os

import pandas as pd
import torch
import torch_em.self_training as self_training
from torch_em.model import UNet2d

import common


def check_loader(args, n_images=5):
    from torch_em.util.debug import check_loader

    cell_types = args.cell_types
    print("The cell types", cell_types, "were selected.")
    print("Checking the unsupervised loader for the first cell type", cell_types[0])

    loader = common.get_unsupervised_loader(
        args, "train", cell_types[0],
        teacher_augmentation="weak", student_augmentation="weak",
    )
    check_loader(loader, n_images)


def _train_source_target(args, source_cell_type, target_cell_type):
    model = UNet2d(in_channels=1, out_channels=1, initial_features=64, final_activation="Sigmoid")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    # self training functionality
    thresh = args.confidence_threshold
    pseudo_labeler = self_training.DefaultPseudoLabeler(confidence_threshold=thresh)
    loss = self_training.DefaultSelfTrainingLoss()
    loss_and_metric = self_training.DefaultSelfTrainingLossAndMetric()

    # data loaders
    supervised_train_loader = common.get_supervised_loader(args, "train", source_cell_type)
    supervised_val_loader = common.get_supervised_loader(args, "val", source_cell_type)
    unsupervised_train_loader = common.get_unsupervised_loader(
        args, "train", target_cell_type,
        teacher_augmentation="weak", student_augmentation="weak",
    )
    unsupervised_val_loader = common.get_unsupervised_loader(
        args, "val", target_cell_type,
        teacher_augmentation="weak", student_augmentation="weak",
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    name = f"unet_adamt/thresh-{thresh}/{source_cell_type}/{target_cell_type}"
    trainer = self_training.MeanTeacherTrainer(
        name=name,
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        pseudo_labeler=pseudo_labeler,
        unsupervised_loss=loss,
        unsupervised_loss_and_metric=loss_and_metric,
        supervised_train_loader=supervised_train_loader,
        unsupervised_train_loader=unsupervised_train_loader,
        supervised_val_loader=supervised_val_loader,
        unsupervised_val_loader=unsupervised_val_loader,
        supervised_loss=loss,
        supervised_loss_and_metric=loss_and_metric,
        logger=self_training.SelfTrainingTensorboardLogger,
        mixed_precision=True,
        device=device,
        log_image_interval=100,
    )
    trainer.fit(args.n_iterations)


def _train_source(args, cell_type):
    for target_cell_type in common.CELL_TYPES:
        if target_cell_type == cell_type:
            continue
        _train_source_target(args, cell_type, target_cell_type)


def run_training(args):
    for cell_type in args.cell_types:
        print("Start training for cell type:", cell_type)
        _train_source(args, cell_type)


def run_evaluation(args):
    results = []
    for ct in args.cell_types:
        res = common.evaluate_transfered_model(args, ct, "unet_adamt", model_state="teacher_state")
        results.append(res)
    results = pd.concat(results)
    print("Evaluation results:")
    print(results)
    result_folder = "./results"
    os.makedirs(result_folder, exist_ok=True)
    results.to_csv(os.path.join(result_folder, "unet_adamt.csv"), index=False)


def main():
    parser = common.get_parser(default_iterations=75000, default_batch_size=4)
    parser.add_argument("--confidence_threshold", default=0.9)
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
