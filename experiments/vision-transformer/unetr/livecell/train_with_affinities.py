import os

import torch

import common


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # overwrite to use complex device setups
    patch_shape = (512, 512)  # patch size used for training on livecell

    # directory folder to save different parts of the scheme
    dir_structure = os.path.join(
        args.model_name, "affinities",
        f"{args.source_choice}-sam" if args.do_sam_ini else f"{args.source_choice}-scratch"
    )

    # get the desired loss function for training
    loss = common.get_loss_function(with_affinities=True)

    # get the model for the training and inference on livecell dataset
    model = common.get_unetr_model(
        model_name=args.model_name, source_choice=args.source_choice, patch_shape=patch_shape,
        sam_initialization=args.do_sam_ini, output_channels=(len(common.OFFSETS) + 1)
    )
    model.to(device)

    # determining where to save the checkpoints and tensorboard logs
    save_root = os.path.join(args.save_root, dir_structure) if args.save_root is not None else args.save_root

    # determines the directory where the predictions will be saved
    root_save_dir = os.path.join(args.save_dir, dir_structure)

    if args.train:
        print("2d UNETR training (with affinities) on LiveCELL...")

        # get the desired livecell loaders for training
        train_loader, val_loader = common.get_my_livecell_loaders(
            args.input, patch_shape, args.cell_type, with_affinities=True,
            # the logic written for `input_norm` is complicated, but the idea is simple:
            # - use default norm for the inputs when we "DONOT" use SAM initialization
            # - use identity trafo for the inputs when we use SAM initialization
            input_norm=not args.do_sam_ini
        )

        common.do_unetr_training(
            train_loader=train_loader, val_loader=val_loader, model=model, loss=loss,
            device=device, save_root=save_root, iterations=args.iterations
        )

    if args.predict:
        print("2d UNETR inference (with affinities) on LiveCELL...")
        common.do_unetr_inference(
            input_path=args.input, device=device, model=model, save_root=save_root,
            root_save_dir=root_save_dir, with_affinities=True,
            # the logic written for `input_norm` is complicated, but the idea is simple:
            # - should standardize the inputs when we "DONOT" use SAM initialization
            # - should not standardize the inputs when we use SAM initialization
            input_norm=not args.do_sam_ini
        )
        print("Predictions are saved in", root_save_dir)

    if args.evaluate:
        print("2d UNETR evaluation (with affinities) on LiveCELL...")
        csv_save_dir = os.path.join("results", dir_structure)
        os.makedirs(csv_save_dir, exist_ok=True)

        common.do_unetr_evaluation(
            input_path=args.input, root_save_dir=root_save_dir, csv_save_dir=csv_save_dir, with_affinities=True
        )


if __name__ == "__main__":
    parser = common.get_parser()
    args = parser.parse_args()
    main(args)
