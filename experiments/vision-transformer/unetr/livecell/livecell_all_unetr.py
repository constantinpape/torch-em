import os
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

import torch
import torch_em

import common


def do_unetr_training(
        train_loader,
        val_loader,
        model,
        device: torch.device,
        iterations: int,
        loss,
        save_root: str
):
    print("Run training for all cell types")
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


def do_unetr_inference(
        input_path: str,
        device: torch.device,
        model,
        root_save_dir: str,
        save_root: str,
        with_affinities: bool
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
            common.predict_for_unetr(img_path, model, root_save_dir, device, with_affinities)


def do_unetr_evaluation(
        input_path: str,
        root_save_dir: str,
        csv_save_dir: str,
        with_affinities: bool
):
    _save_dir = os.path.join(root_save_dir, "src-all")
    assert os.path.exists(_save_dir), _save_dir

    gt_dir = os.path.join(input_path, "annotations", "livecell_test_images", "*", "*")

    mws_msa_list, mws_sa50_list = [], []
    fg_list, bd_list, msa1_list, sa501_list, msa2_list, sa502_list = [], [], [], [], [], []
    for gt_path in tqdm(glob(gt_dir)):
        all_metrics = common.evaluate_for_unetr(gt_path, _save_dir, with_affinities)
        if with_affinities:
            msa, sa50 = all_metrics
            mws_msa_list.append(msa)
            mws_sa50_list.append(sa50)
        else:
            fg_dice, bd_dice, msa1, sa_acc1, msa2, sa_acc2 = all_metrics
            fg_list.append(fg_dice)
            bd_list.append(bd_dice)
            msa1_list.append(msa1)
            sa501_list.append(sa_acc1[0])
            msa2_list.append(msa2)
            sa502_list.append(sa_acc2[0])

    if with_affinities:
        res_dict = {
            "LIVECell": "Metrics",
            "mSA": np.mean(mws_msa_list),
            "SA50": np.mean(mws_sa50_list)
        }
    else:
        res_dict = {
            "LIVECell": "Metrics",
            "ws1_mSA": np.mean(msa1_list),
            "ws1_SA50": np.mean(sa501_list),
            "ws2_mSA": np.mean(msa2_list),
            "ws2_SA50": np.mean(sa502_list)
        }

    df = pd.DataFrame.from_dict([res_dict])
    df.to_csv(os.path.join(csv_save_dir, "livecell.csv"))


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # overwrite to use complex device setups
    patch_shape = (512, 512)  # patch size used for training on livecell

    # directory folder to save different parts of the scheme
    dir_structure = os.path.join(
        args.model_name, "affinities" if args.with_affinities else "boundaries",
        f"{args.source_choice}-sam" if args.do_sam_ini else f"{args.source_choice}-scratch"
    )

    # get the desired loss function for training
    loss = common.get_loss_function(
        with_affinities=args.with_affinities  # takes care of calling the loss for training with affinities
    )

    # get the model for the training and inference on livecell dataset
    model = common.get_unetr_model(
        model_name=args.model_name, source_choice=args.source_choice, patch_shape=patch_shape,
        sam_initialization=args.do_sam_ini, output_channels=common._get_output_channels(args.with_affinities),
        backbone=args.pretrained_choice
    )
    model.to(device)

    # determining where to save the checkpoints and tensorboard logs
    save_root = os.path.join(args.save_root, dir_structure) if args.save_root is not None else args.save_root

    if args.train:
        print("2d UNETR training on LIVECell dataset")
        # get the desired livecell loaders for training
        train_loader, val_loader = common.get_my_livecell_loaders(
            args.input, patch_shape, args.cell_type, with_boundary=not args.with_affinities,
            with_affinities=args.with_affinities,  # this takes care of getting the loaders with affinities
            no_input_norm=args.do_sam_ini  # if sam ini, use identity raw trafo, else use default raw trafo
        )
        do_unetr_training(
            train_loader=train_loader, val_loader=val_loader, model=model,
            device=device, save_root=save_root, iterations=args.iterations, loss=loss
        )

    # determines the directory where the predictions will be saved
    root_save_dir = os.path.join(args.save_dir, dir_structure)

    if args.predict:
        print("2d UNETR inference on LIVECell dataset")
        do_unetr_inference(
            input_path=args.input, device=device, model=model, save_root=save_root,
            root_save_dir=root_save_dir, with_affinities=args.with_affinities
        )
        print("Predictions are saved in", root_save_dir)

    if args.evaluate:
        print("2d UNETR evaluation on LIVECell dataset")
        csv_save_dir = os.path.join("results", dir_structure)
        os.makedirs(csv_save_dir, exist_ok=True)

        do_unetr_evaluation(
            input_path=args.input, root_save_dir=root_save_dir,
            csv_save_dir=csv_save_dir, with_affinities=args.with_affinities
        )


if __name__ == "__main__":
    parser = common.get_parser()
    args = parser.parse_args()
    main(args)
