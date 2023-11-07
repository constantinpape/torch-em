import os
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from typing import List

import torch
import torch_em

import common


def do_unetr_training(
        train_loader,
        val_loader,
        model,
        cell_types: List[str],
        device: torch.device,
        iterations: int,
        loss,
        save_root: str
):
    print("Run training for cell types:", cell_types)
    trainer = torch_em.default_segmentation_trainer(
        name=f"livecell-{cell_types}",
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
        cell_types: List[str],
        root_save_dir: str,
        save_root: str,
        with_affinities: bool
):
    test_img_dir = os.path.join(input_path, "images", "livecell_test_images", "*")
    for ctype in cell_types:
        model_ckpt = os.path.join(save_root, "checkpoints", f"livecell-{ctype}", "best.pt")
        assert os.path.exists(model_ckpt), model_ckpt

        model.load_state_dict(torch.load(model_ckpt, map_location=torch.device('cpu'))["model_state"])
        model.to(device)
        model.eval()

        # creating the respective directories for saving the outputs
        os.makedirs(os.path.join(root_save_dir, f"src-{ctype}"), exist_ok=True)

        with torch.no_grad():
            for img_path in tqdm(glob(test_img_dir), desc=f"Run inference for all livecell with model {model_ckpt}"):
                common.predict_for_unetr(img_path, model, root_save_dir, ctype, device, with_affinities)


def do_unetr_evaluation(
        input_path: str,
        cell_types: List[str],
        root_save_dir: str,
        csv_save_dir: str,
        with_affinities: bool
):
    # list for foreground-boundary evaluations
    fg_list, bd_list = [], []
    ws1_msa_list, ws2_msa_list, ws1_sa50_list, ws2_sa50_list = [], [], [], []

    # lists for affinities evaluation
    mws_msa_list, mws_sa50_list = [], []

    for c1 in cell_types:
        # we check whether we have predictions from a particular cell-type
        _save_dir = os.path.join(root_save_dir, f"src-{c1}")
        if not os.path.exists(_save_dir):
            print("Skipping", _save_dir)
            continue

        # dict for foreground-boundary evaluations
        fg_set, bd_set = {"CELL TYPE": c1}, {"CELL TYPE": c1}
        (ws1_msa_set, ws2_msa_set,
         ws1_sa50_set, ws2_sa50_set) = {"CELL TYPE": c1}, {"CELL TYPE": c1}, {"CELL TYPE": c1}, {"CELL TYPE": c1}

        # dict for affinities evaluation
        mws_msa_set, mws_sa50_set = {"CELL TYPE": c1}, {"CELL TYPE": c1}

        for c2 in tqdm(cell_types, desc=f"Evaluation on {c1} source models from {_save_dir}"):
            gt_dir = os.path.join(input_path, "annotations", "livecell_test_images", c2, "*")

            # cell-wise evaluation list for foreground-boundary evaluations
            cwise_fg, cwise_bd = [], []
            cwise_ws1_msa, cwise_ws2_msa, cwise_ws1_sa50, cwise_ws2_sa50 = [], [], [], []

            # cell-wise evaluation list for affinities evaluation
            cwise_mws_msa, cwise_mws_sa50 = [], []

            for gt_path in glob(gt_dir):
                all_metrics = common.evaluate_for_unetr(gt_path, _save_dir, with_affinities)
                if with_affinities:
                    mws_msa, mws_sa50 = all_metrics
                    cwise_mws_msa.append(mws_msa)
                    cwise_mws_sa50.append(mws_sa50)
                else:
                    fg_dice, bd_dice, msa1, sa_acc1, msa2, sa_acc2 = all_metrics
                    cwise_fg.append(fg_dice)
                    cwise_bd.append(bd_dice)
                    cwise_ws1_msa.append(msa1)
                    cwise_ws2_msa.append(msa2)
                    cwise_ws1_sa50.append(sa_acc1[0])
                    cwise_ws2_sa50.append(sa_acc2[0])

            if with_affinities:
                mws_msa_set[c2], mws_sa50_set[c2] = np.mean(cwise_mws_msa), np.mean(cwise_mws_sa50)
            else:
                fg_set[c2], bd_set[c2] = np.mean(cwise_fg), np.mean(cwise_bd)
                ws1_msa_set[c2], ws2_msa_set[c2] = np.mean(cwise_ws1_msa), np.mean(cwise_ws2_msa)
                ws1_sa50_set[c2], ws2_sa50_set[c2] = np.mean(cwise_ws1_sa50), np.mean(cwise_ws2_sa50)

        if with_affinities:
            mws_msa_list.append(pd.DataFrame.from_dict([mws_msa_set]))
            mws_sa50_list.append(pd.DataFrame.from_dict([mws_sa50_set]))
        else:
            fg_list.append(pd.DataFrame.from_dict([fg_set]))
            bd_list.append(pd.DataFrame.from_dict([bd_set]))
            ws1_msa_list.append(pd.DataFrame.from_dict([ws1_msa_set]))
            ws2_msa_list.append(pd.DataFrame.from_dict([ws2_msa_set]))
            ws1_sa50_list.append(pd.DataFrame.from_dict([ws1_sa50_set]))
            ws2_sa50_list.append(pd.DataFrame.from_dict([ws2_sa50_set]))

    if with_affinities:
        df_mws_msa, df_mws_sa50 = pd.concat(mws_msa_list, ignore_index=True), pd.concat(mws_sa50_list, ignore_index=True)
        df_mws_msa.to_csv(os.path.join(csv_save_dir, "mws-affs-msa.csv"))
        df_mws_sa50.to_csv(os.path.join(csv_save_dir, "mws-affs-sa50.csv"))
    else:
        df_fg, df_bd = pd.concat(fg_list, ignore_index=True), pd.concat(bd_list, ignore_index=True)
        df_ws1_msa, df_ws2_msa = pd.concat(ws1_msa_list, ignore_index=True), pd.concat(ws2_msa_list, ignore_index=True)
        df_ws1_sa50, df_ws2_sa50 = pd.concat(ws1_sa50_list, ignore_index=True), pd.concat(ws2_sa50_list, ignore_index=True)
        df_fg.to_csv(os.path.join(csv_save_dir, "foreground-dice.csv"))
        df_bd.to_csv(os.path.join(csv_save_dir, "boundary-dice.csv"))
        df_ws1_msa.to_csv(os.path.join(csv_save_dir, "watershed1-msa.csv"))
        df_ws2_msa.to_csv(os.path.join(csv_save_dir, "watershed2-msa.csv"))
        df_ws1_sa50.to_csv(os.path.join(csv_save_dir, "watershed1-sa50.csv"))
        df_ws2_sa50.to_csv(os.path.join(csv_save_dir, "watershed2-sa50.csv"))


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # overwrite to use complex device setups
    patch_shape = (512, 512)  # patch size used for training on livecell

    # get the desired loss function for training
    loss = common.get_loss_function(
        with_affinities=args.with_affinities  # takes care of calling the loss for training with affinities
    )

    # get the model for the training and inference on livecell dataset
    model = common.get_unetr_model(
        model_name=args.model_name, source_choice=args.source_choice, patch_shape=patch_shape,
        sam_initialization=args.do_sam_ini, output_channels=common._get_output_channels(args.with_affinities)
    )
    model.to(device)

    # determining where to save the checkpoints and tensorboard logs
    save_root = os.path.join(
        args.save_root, "affinities" if args.with_affinities else "boundaries",
        f"{args.source_choice}-sam" if args.do_sam_ini else f"{args.source_choice}-scratch"
    ) if args.save_root is not None else args.save_root

    if args.train:
        print("2d UNETR training on LIVECell dataset")
        # get the desired livecell loaders for training
        train_loader, val_loader = common.get_my_livecell_loaders(
            args.input, patch_shape, args.cell_type,
            with_affinities=args.with_affinities  # this takes care of getting the loaders with affinities
        )
        do_unetr_training(
            train_loader=train_loader, val_loader=val_loader, model=model, cell_types=args.cell_type,
            device=device, save_root=save_root, iterations=args.iterations, loss=loss
        )

    # determines the directory where the predictions will be saved
    root_save_dir = os.path.join(
        args.save_dir, "affinities" if args.with_affinities else "boundaries",
        f"{args.source_choice}-sam" if args.do_sam_ini else f"{args.source_choice}-scratch"
    )

    if args.predict:
        print("2d UNETR inference on LIVECell dataset")
        do_unetr_inference(
            input_path=args.input, device=device, model=model, cell_types=common.CELL_TYPES,
            root_save_dir=root_save_dir, save_root=save_root, with_affinities=args.with_affinities
        )
        print("Predictions are saved in", root_save_dir)

    if args.evaluate:
        print("2d UNETR evaluation on LIVECell dataset")
        tmp_csv_name = f"{args.source_choice}-sam" if args.do_sam_ini else f"{args.source_choice}-scratch"
        csv_save_dir = os.path.join("results", "affinities" if args.with_affinities else "boundaries", tmp_csv_name)
        os.makedirs(csv_save_dir, exist_ok=True)

        do_unetr_evaluation(
            input_path=args.input, cell_types=common.CELL_TYPES, root_save_dir=root_save_dir,
            csv_save_dir=csv_save_dir, with_affinities=args.with_affinities
        )


if __name__ == "__main__":
    parser = common.get_parser()
    args = parser.parse_args()
    main(args)
