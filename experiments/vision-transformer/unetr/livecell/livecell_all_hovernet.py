import os
from tqdm import tqdm
from glob import glob

import numpy as np
import pandas as pd

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


def do_unetr_hovernet_inference(
        input_path: str,
        device: torch.device,
        model,
        root_save_dir: str,
        save_root: str,
        with_distance_maps: bool
):
    test_img_dir = os.path.join(input_path, "images", "livecell_test_images", "*")
    model_ckpt = os.path.join(save_root, "checkpoints", "livecell-all", "best.pt")
    assert os.path.exists(model_ckpt), model_ckpt

    model.load_state_dict(torch.load(model_ckpt, map_location=torch.device('cpu'))["model_state"])
    model.to(device)
    model.eval()

    # creating the respective directories for saving the outputs
    os.makedirs(os.path.join(root_save_dir, "src-all"), exist_ok=True)

    with torch.no_grad():
        for img_path in tqdm(glob(test_img_dir), desc=f"Run inference for all livecell with model {model_ckpt}"):
            common.predict_for_unetr(img_path, model, root_save_dir, device, with_distance_maps=with_distance_maps)


def do_unetr_hovernet_evaluation(
        input_path: str,
        root_save_dir: str,
        csv_save_dir: str,
        with_distance_maps: bool
):
    _save_dir = os.path.join(root_save_dir, "src-all")
    assert os.path.exists(_save_dir), _save_dir

    gt_dir = os.path.join(input_path, "annotations", "livecell_test_images", "*", "*")

    msa_list, sa50_list = [], []
    for gt_path in tqdm(glob(gt_dir)):
        all_metrics = common.evaluate_for_unetr(gt_path, _save_dir, with_distance_maps=with_distance_maps)
        msa, sa50 = all_metrics
        msa_list.append(msa)
        sa50_list.append(sa50)

    res_dict = {
        "LIVECell": "Metrics",
        "mSA": np.mean(msa_list),
        "SA50": np.mean(sa50_list)
    }

    df = pd.DataFrame.from_dict([res_dict])
    df.to_csv(os.path.join(csv_save_dir, "livecell.csv"))


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # overwrite to use complex device setups
    patch_shape = (512, 512)  # patch size used for training on livecell

    # directory folder to save different parts of the scheme
    dir_structure = os.path.join(args.model_name, "hovernet", "torch-em-sam")

    # get the desired loss function for training
    loss = common.get_loss_function(with_distance_maps=True)

    # get the model for the training and inference on livecell dataset
    model = common.get_unetr_model(
        model_name=args.model_name, source_choice="torch-em", patch_shape=patch_shape,
        sam_initialization=args.do_sam_ini, output_channels=3,  # foreground-background, x-map, y-map
        backbone=args.pretrained_choice
    )
    model.to(device)

    # determining where to save the checkpoints and tensorboard logs
    save_root = os.path.join(args.save_root, dir_structure) if args.save_root is not None else args.save_root

    if args.train:
        print("2d UNETR hovernet-style training on LIVECell dataset")
        # get the desried livecell loaders for training
        train_loader, val_loader = common.get_my_livecell_loaders(
            args.input, patch_shape, args.cell_type, with_distance_maps=True
        )
        do_unetr_hovernet_training(
            train_loader=train_loader, val_loader=val_loader, model=model,
            device=device, save_root=save_root, iterations=args.iterations, loss=loss
        )

    # determines the directory where the predictions will be saved
    root_save_dir = os.path.join(args.save_dir, dir_structure)

    if args.predict:
        print("2d UNETR hovernet-style inference on LIVECell dataset")
        do_unetr_hovernet_inference(
            input_path=args.input, device=device, model=model, save_root=save_root,
            root_save_dir=root_save_dir, with_distance_maps=True
        )
        print("Predictions are saved in", root_save_dir)

    if args.evaluate:
        print("2d UNETR hovernet-style evaluation on LIVECell dataset")
        csv_save_dir = os.path.join("results", dir_structure)
        os.makedirs(csv_save_dir, exist_ok=True)

        do_unetr_hovernet_evaluation(
            input_path=args.input, root_save_dir=root_save_dir,
            csv_save_dir=csv_save_dir, with_distance_maps=True
        )


if __name__ == "__main__":
    parser = common.get_parser()
    args = parser.parse_args()
    main(args)
