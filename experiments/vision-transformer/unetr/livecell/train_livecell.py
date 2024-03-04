import os

import torch

import common


def main(args):
    assert args.experiment_name in ["boundaries", "affinities", "distances"], args.experiment_name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # overwrite to use complex device setups
    patch_shape = tuple(args.patch_shape)  # patch size used for training on livecell

    _name = args.model_name if not args.use_unet else "unet"

    # directory folder to save different parts of the scheme
    dir_structure = os.path.join(
        f"{args.source_choice}-sam" if args.do_sam_ini else f"{args.source_choice}-scratch",
        _name, args.experiment_name
    )

    # get the desired loss function for training
    loss = common.get_loss_function(args.experiment_name)

    if args.use_unet:
        model = common.get_unet_model(output_channels=common.get_output_channels(args.experiment_name))
        _store_model_name = "unet"
    else:
        # get the unetr model for the training and inference on livecell dataset
        model = common.get_unetr_model(
            model_name=args.model_name,
            source_choice=args.source_choice,
            patch_shape=patch_shape,
            sam_initialization=args.do_sam_ini,
            output_channels=common.get_output_channels(args.experiment_name)
        )
        _store_model_name = "unetr"
    model.to(device)

    # determining where to save the checkpoints and tensorboard logs
    save_root = os.path.join(args.save_root, dir_structure) if args.save_root is not None else args.save_root

    # determines the directory where the predictions will be saved
    root_save_dir = os.path.join(save_root, "inference")

    if args.train:
        print(f"2d {_store_model_name.upper()} training (with {args.experiment_name}) on LiveCELL...")

        # get the desired livecell loaders for training
        train_loader, val_loader = common.get_my_livecell_loaders(
            args.input, patch_shape, experiment_name=args.experiment_name,
            # the logic written for `input_norm` is complicated, but the idea is simple:
            # - use default norm for the inputs when we "DONOT" use SAM initialization
            # - use identity trafo for the inputs when we use SAM initialization
            input_norm=not args.do_sam_ini
        )

        common.do_unetr_training(
            train_loader=train_loader, val_loader=val_loader, model=model, loss=loss,
            device=device, save_root=save_root, iterations=args.iterations, name=f"livecell-{_store_model_name}"
        )

    if args.predict:
        print(f"2d {_store_model_name.upper()} inference (with {args.experiment_name}) on LiveCELL...")
        common.do_unetr_inference(
            input_path=args.input, device=device, model=model, save_root=save_root,
            root_save_dir=root_save_dir, experiment_name=args.experiment_name,
            input_norm=not args.do_sam_ini, name_extension=f"livecell-{_store_model_name}"
        )
        print("Predictions are saved in", root_save_dir)

    if args.evaluate:
        print(f"2d {_store_model_name.upper()} evaluation (with {args.experiment_name}) on LiveCELL...")
        csv_save_dir = os.path.join("results", dir_structure)
        os.makedirs(csv_save_dir, exist_ok=True)

        common.do_unetr_evaluation(
            input_path=args.input, root_save_dir=root_save_dir,
            csv_save_dir=csv_save_dir, experiment_name=args.experiment_name
        )


if __name__ == "__main__":
    parser = common.get_parser()
    args = parser.parse_args()
    main(args)
