import os
from collections import OrderedDict

import torch
from torch_em import model as torch_em_models

import common


def prune_prefix(checkpoint_path):
    state = torch.load(checkpoint_path, map_location="cpu")
    model_state = state["model_state"]

    # let's prune the `.sam` prefix for the finetuned models
    sam_prefix = "sam.image_encoder."
    updated_model_state = []
    for k, v in model_state.items():
        if k.startswith(sam_prefix):
            updated_model_state.append((k[len(sam_prefix):], v))
    updated_model_state = OrderedDict(updated_model_state)

    return updated_model_state


def get_custom_unetr_model(
        device, model_name, sam_initialization, output_channels, checkpoint_path, freeze_encoder, joint_training
):
    if checkpoint_path is not None:
        if checkpoint_path.endswith("pt"):  # for finetuned models
            model_state = prune_prefix(checkpoint_path)
        else:  # for vanilla sam models
            model_state = checkpoint_path
    else:  # while checkpoint path is None, hence we train from scratch
        model_state = checkpoint_path

    model = torch_em_models.UNETR(
        backbone="sam",
        encoder=model_name,
        out_channels=output_channels,
        use_sam_stats=sam_initialization,
        final_activation="Sigmoid",
        encoder_checkpoint=model_state,
        use_skip_connection=not joint_training  # if joint_training, no skip con. else, use skip con. by default
    )

    model.to(device)

    # if expected, let's freeze the image encoder
    if freeze_encoder:
        for name, param in model.named_parameters():
            if name.startswith("encoder"):
                param.requires_grad = False

    return model


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # overwrite to use complex device setups
    patch_shape = (512, 512)  # patch size used for training on livecell

    # directory folder to save different parts of the scheme
    dir_structure = os.path.join(
        args.model_name, f"freeze_encoder_{args.freeze_encoder}", "distances", "dicebaseddistloss",
        f"{args.source_choice}-sam" if args.do_sam_ini else f"{args.source_choice}-scratch"
    )

    # get the desired loss function for training
    loss = common.get_loss_function(with_distances=True, combine_dist_with_dice=True)

    # get the custom model for the training and inference on livecell dataset
    model = get_custom_unetr_model(
        device, args.model_name, sam_initialization=args.do_sam_ini, output_channels=3,
        checkpoint_path=args.checkpoint, freeze_encoder=args.freeze_encoder, joint_training=args.joint_training
    )

    # determining where to save the checkpoints and tensorboard logs
    save_root = os.path.join(args.save_root, dir_structure) if args.save_root is not None else args.save_root

    # determines the directory where the predictions will be saved
    root_save_dir = os.path.join(args.save_dir, dir_structure)

    if args.train:
        print("2d (custom) UNETR training (with distances) on LiveCELL...")

        # get the desired livecell loaders for training
        train_loader, val_loader = common.get_my_livecell_loaders(
            args.input, patch_shape, args.cell_type, with_distances=True,
            input_norm=not args.do_sam_ini
        )

        common.do_unetr_training(
            train_loader=train_loader, val_loader=val_loader, model=model, loss=loss,
            device=device, save_root=save_root, iterations=args.iterations
        )

    if args.predict:
        print("2d (custom) UNETR inference (with distances) on LiveCELL...")
        common.do_unetr_inference(
            input_path=args.input, device=device, model=model, save_root=save_root,
            root_save_dir=root_save_dir, with_distances=True,
            # the logic written for `input_norm` is complicated, but the idea is simple:
            # - should standardize the inputs when we "DONOT" use SAM initialization
            # - should not standardize the inputs when we use SAM initialization
            input_norm=not args.do_sam_ini
        )
        print("Predictions are saved in", root_save_dir)

    if args.evaluate:
        print("2d (custom) UNETR evaluation (with distances) on LiveCELL...")
        csv_save_dir = os.path.join("results", dir_structure)
        os.makedirs(csv_save_dir, exist_ok=True)

        common.do_unetr_evaluation(
            input_path=args.input, root_save_dir=root_save_dir, csv_save_dir=csv_save_dir, with_distances=True
        )


# we train three setups:
#    - training from scratch, seeing the performance using instance segmentation
#    - training from vanilla SAM, seeing the performance using instance segmentation
#    - training from finetuned SAM, seeing the performance using instance segmentation
if __name__ == "__main__":
    parser = common.get_parser()
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="The checkpoint to the specific pretrained models."
    )
    parser.add_argument(
        "--freeze_encoder", action="store_true", help="Experiments to freeze the encoder."
    )
    parser.add_argument(
        "--joint_training", action="store_true", help="Uses VNETR for training"
    )
    args = parser.parse_args()
    main(args)
