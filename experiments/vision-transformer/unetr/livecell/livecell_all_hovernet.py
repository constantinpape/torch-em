import os

import torch
import torch_em

import common


def do_unetr_hovernet_training(
        train_loader, val_loader, model, device, iterations, loss, save_root
):
    print("Run training with hovernet ideas for all cell types")
    trainer = torch_em.default_segmentation_trainer(
        name="livecell-all",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=1e-5,
        device=device,
        mixed_precision=True,
        log_image_interval=50,
        compile_model=False,
        save_root=save_root,
        loss=loss,
        metric=loss
    )
    trainer.fit(iterations)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # overwrite to use complex device setups
    patch_shape = (512, 512)  # patch size used for training on livecell

    train_loader, val_loader = common.get_my_livecell_loaders(
        args.input, patch_shape, args.cell_type, with_distance_maps=True
    )

    from torch_em.util.debug import check_loader
    check_loader(train_loader, 8, True, True, False, "livecell_train.png")

    breakpoint()

    quit()

    # directory folder to save different parts of the scheme
    dir_structure = os.path.join(args.model_name, "hovernet", "torch-em-sam")

    # get the desired loss function for training
    # TODO: curate the loss function for us
    loss = None

    # get the model for the training and inference on livecell dataset
    model = common.get_unetr_model(
        model_name=args.model_name, source_choice="torch-em", patch_shape=patch_shape, sam_initialization=True,
        output_channels=3  # foreground-background, x-map, y-map
    )
    model.to(device)

    # determining where to save the checkpoints and tensorboard logs
    save_root = os.path.join(args.save_root, dir_structure) if args.save_root is not None else args.save_root

    if args.train:
        print("2d UNETR hovernet-idea training on LIVECell dataset")
        # get the desried livecell loaders for training
        train_loader, val_loader = common.get_my_livecell_loaders(
            args.input, patch_shape, args.cell_type  # TODO: pass label transform to make the transform
        )
        do_unetr_hovernet_training(
            train_loader=train_loader, val_loader=val_loader, model=model,
            device=device, save_root=save_root, iterations=args.iterations, loss=loss
        )


if __name__ == "__main__":
    parser = common.get_parser()
    args = parser.parse_args()
    main(args)
