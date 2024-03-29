import os

import pandas as pd
import torch
import torch_em.self_training as self_training
from torch_em.util import load_model

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

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = common.get_punet()
    if args.save_root is None:
        src_checkpoint = f"./checkpoints/punet_source/{source_cell_type}"
    else:
        src_checkpoint = args.save_root + f"checkpoints/punet_source/{source_cell_type}"
    model = load_model(checkpoint=src_checkpoint, model=model, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.9, patience=10)

    # self training functionality
    # - when thresh is None, we don't mask the reconstruction loss (RL) with label filters
    # - when thresh is passed (float), we mask the RL with weighted consensus label filters
    # - when thresh is passed (float) with args.consensus_masking, we mask the RL with masked consensus label filters
    thresh = args.confidence_threshold
    if thresh is None:
        assert args.consensus_masking is False, "Provide a confidence threshold to use consensus masking"

    pseudo_labeler = self_training.ProbabilisticPseudoLabeler(activation=torch.nn.Sigmoid(),
                                                              confidence_threshold=thresh, prior_samples=16,
                                                              consensus_masking=args.consensus_masking)
    loss = self_training.ProbabilisticUNetLoss()
    loss_and_metric = self_training.ProbabilisticUNetLossAndMetric()

    # data loaders
    unsupervised_train_loader = common.get_unsupervised_loader(
        args, args.batch_size, "train", target_cell_type,
        teacher_augmentation="weak", student_augmentation="weak",
    )
    unsupervised_val_loader = common.get_unsupervised_loader(
        args, 1, "val", target_cell_type,
        teacher_augmentation="weak", student_augmentation="weak",
    )

    if args.consensus_masking:
        name = f"punet_mean_teacher/thresh-{thresh}-masking/{source_cell_type}/{target_cell_type}"
    else:
        name = f"punet_mean_teacher/thresh-{thresh}/{source_cell_type}/{target_cell_type}"

    trainer = self_training.MeanTeacherTrainer(
        name=name,
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        pseudo_labeler=pseudo_labeler,
        unsupervised_loss=loss,
        unsupervised_loss_and_metric=loss_and_metric,
        unsupervised_train_loader=unsupervised_train_loader,
        unsupervised_val_loader=unsupervised_val_loader,
        supervised_loss=loss,
        supervised_loss_and_metric=loss_and_metric,
        logger=None,
        mixed_precision=True,
        device=device,
        log_image_interval=100,
        save_root=args.save_root,
        reinit_teacher=False,
        compile_model=False
    )
    trainer.fit(args.n_iterations)


def _train_source(args, cell_type):
    if args.target_ct is None:
        target_cell_list = common.CELL_TYPES
    else:
        target_cell_list = args.target_ct

    for target_cell_type in target_cell_list:
        print("Training on target cell type:", target_cell_type)
        if target_cell_type == cell_type:
            continue
        _train_source_target(args, cell_type, target_cell_type)


def run_training(args):
    for cell_type in args.cell_types:
        print("Start training for source cell type:", cell_type)
        _train_source(args, cell_type)


def run_evaluation(args):
    results = []
    for ct in args.cell_types:
        res = common.evaluate_transfered_model(args, ct, "punet_mean_teacher",
                                               get_model=common.get_punet,
                                               model_state="teacher_state",
                                               prediction_function=common.get_punet_predictions)
        results.append(res)
    results = pd.concat(results)
    print("Evaluation results:")
    print(results)
    result_folder = "./results"
    os.makedirs(result_folder, exist_ok=True)
    results.to_csv(os.path.join(result_folder, "punet_mean_teacher.csv"), index=False)


def main():
    parser = common.get_parser(default_iterations=10000, default_batch_size=4)
    parser.add_argument("--confidence_threshold", default=None, type=float)
    parser.add_argument("--consensus_masking", action='store_true')
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
