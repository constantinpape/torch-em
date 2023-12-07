import os

import torch

import common


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # overwrite to use complex device setups
    patch_shape = (512, 512)  # patch size used for training on livecell

    # directory folder to save different parts of the scheme
    dir_structure = os.path.join(
        args.model_name, "distances",
        f"{args.source_choice}-sam" if args.do_sam_ini else f"{args.source_choice}-scratch"
    )

    # get the desired loss function for training
    loss = common.get_loss_function(with_distances=True, combine_dist_with_dice=args.with_dice)

    # get the model for the training and inference on livecell dataset
    model = common.get_unetr_model(
        model_name=args.model_name, source_choice=args.source_choice, patch_shape=patch_shape,
        sam_initialization=args.do_sam_ini, output_channels=3
    )
    model.to(device)

    # determining where to save the checkpoints and tensorboard logs
    save_root = os.path.join(args.save_root, dir_structure) if args.save_root is not None else args.save_root

    # determines the directory where the predictions will be saved
    root_save_dir = os.path.join(args.save_dir, dir_structure)

    if args.train:
        print("2d UNETR training (with distances) on LiveCELL...")

        # get the desired livecell loaders for training
        train_loader, val_loader = common.get_my_livecell_loaders(
            args.input, patch_shape, args.cell_type, with_distances=True,
            no_input_norm=args.do_sam_ini  # if sam ini, use identity raw trafo, else use default raw trafo
        )

        common.do_unetr_training(
            train_loader=train_loader, val_loader=val_loader, model=model, loss=loss,
            device=device, save_root=save_root, iterations=args.iterations
        )

    if args.predict:
        print("2d UNETR inference (with distances) on LiveCELL...")
        common.do_unetr_inference(
            input_path=args.input, device=device, model=model, save_root=save_root,
            root_save_dir=root_save_dir, with_distances=True
        )
        print("Predictions are saved in", root_save_dir)

    if args.evaluate:
        print("2d UNETR evaluation (with distances) on LiveCELL...")
        csv_save_dir = os.path.join("results", dir_structure)
        os.makedirs(csv_save_dir, exist_ok=True)

        common.do_unetr_evaluation(
            input_path=args.input, root_save_dir=root_save_dir, csv_save_dir=csv_save_dir, with_distances=True
        )


if __name__ == "__main__":
    parser = common.get_parser()
    parser.add_argument("--with_dice", action="store_true", help="Uses dice loss for the distances as well")
    args = parser.parse_args()
    main(args)
