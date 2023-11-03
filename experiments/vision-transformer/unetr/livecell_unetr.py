import os
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from typing import List

import imageio.v2 as imageio
from skimage.segmentation import find_boundaries
from elf.evaluation import dice_score, mean_segmentation_accuracy

import torch
import torch_em
from torch_em.util import segmentation
from torch_em.transform.raw import standardize
from torch_em.util.prediction import predict_with_halo

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
    for ctype in cell_types:
        test_img_dir = os.path.join(input_path, "images", "livecell_test_images", "*")

        model_ckpt = os.path.join(save_root, "checkpoints", f"livecell-{ctype}", "best.pt")
        assert os.path.exists(model_ckpt)

        model.load_state_dict(torch.load(model_ckpt, map_location=torch.device('cpu'))["model_state"])
        model.to(device)
        model.eval()

        # creating the respective directories for saving the outputs
        _settings = ["foreground", "boundary", "watershed1", "watershed2"]
        for _setting in _settings:
            tmp_save_dir = os.path.join(root_save_dir, f"src-{ctype}", _setting)
            os.makedirs(tmp_save_dir, exist_ok=True)

        with torch.no_grad():
            for img_path in tqdm(glob(test_img_dir), desc=f"Run inference for all livecell with model {model_ckpt}"):
                fname = os.path.split(img_path)[-1]

                input_img = imageio.imread(img_path)
                input_img = standardize(input_img)

                if with_affinities:
                    raise NotImplementedError("This still needs to be implemented for affinity-based training")

                else:  # inference using foreground-boundary inputs - for the unetr training
                    outputs = predict_with_halo(
                        input_img, model, gpu_ids=[device], block_shape=[384, 384], halo=[64, 64], disable_tqdm=True
                    )
                    fg, bd = outputs[0, :, :], outputs[1, :, :]
                    ws1 = segmentation.watershed_from_components(bd, fg, min_size=10)
                    ws2 = segmentation.watershed_from_maxima(bd, fg, min_size=10, min_distance=1)

                    _save_outputs = [fg, bd, ws1, ws2]
                    for _setting, _output in zip(_settings, _save_outputs):
                        imageio.imwrite(os.path.join(root_save_dir, f"src-{ctype}", _setting, fname), _output)


def do_unetr_evaluation(
        input_path: str,
        cell_types: List[str],
        root_save_dir: str,
        sam_initialization: bool,
        source_choice: str
):
    fg_list, bd_list = [], []
    ws1_msa_list, ws2_msa_list, ws1_sa50_list, ws2_sa50_list = [], [], [], []

    for c1 in cell_types:
        _save_dir = os.path.join(root_save_dir, f"src-{c1}")
        if not os.path.exists(_save_dir):
            print("Skipping", _save_dir)
            continue

        fg_set, bd_set = {"CELL TYPE": c1}, {"CELL TYPE": c1}
        (ws1_msa_set, ws2_msa_set,
         ws1_sa50_set, ws2_sa50_set) = {"CELL TYPE": c1}, {"CELL TYPE": c1}, {"CELL TYPE": c1}, {"CELL TYPE": c1}
        for c2 in tqdm(cell_types, desc=f"Evaluation on {c1} source models from {_save_dir}"):
            fg_dir = os.path.join(_save_dir, "foreground")
            bd_dir = os.path.join(_save_dir, "boundary")
            ws1_dir = os.path.join(_save_dir, "watershed1")
            ws2_dir = os.path.join(_save_dir, "watershed2")

            gt_dir = os.path.join(input_path, "annotations", "livecell_test_images", c2, "*")
            cwise_fg, cwise_bd = [], []
            cwise_ws1_msa, cwise_ws2_msa, cwise_ws1_sa50, cwise_ws2_sa50 = [], [], [], []
            for gt_path in glob(gt_dir):
                fname = os.path.split(gt_path)[-1]

                gt = imageio.imread(gt_path)
                fg = imageio.imread(os.path.join(fg_dir, fname))
                bd = imageio.imread(os.path.join(bd_dir, fname))
                ws1 = imageio.imread(os.path.join(ws1_dir, fname))
                ws2 = imageio.imread(os.path.join(ws2_dir, fname))

                true_bd = find_boundaries(gt)

                # Compare the foreground prediction to the ground-truth.
                # Here, it's important not to threshold the segmentation. Otherwise EVERYTHING will be set to
                # foreground in the dice function, since we have a comparision > 0 in there, and everything in the
                # binary prediction evaluates to true.
                # For the GT we can set the threshold to 0, because this will map to the correct binary mask.
                cwise_fg.append(dice_score(fg, gt, threshold_gt=0, threshold_seg=None))

                # Compare the background prediction to the ground-truth.
                # Here, we don't need any thresholds: for the prediction the same holds as before.
                # For the ground-truth we have already a binary label, so we don't need to threshold it again.
                cwise_bd.append(dice_score(bd, true_bd, threshold_gt=None, threshold_seg=None))

                msa1, sa_acc1 = mean_segmentation_accuracy(ws1, gt, return_accuracies=True)  # type: ignore
                msa2, sa_acc2 = mean_segmentation_accuracy(ws2, gt, return_accuracies=True)  # type: ignore

                cwise_ws1_msa.append(msa1)
                cwise_ws2_msa.append(msa2)
                cwise_ws1_sa50.append(sa_acc1[0])
                cwise_ws2_sa50.append(sa_acc2[0])

            fg_set[c2] = np.mean(cwise_fg)
            bd_set[c2] = np.mean(cwise_bd)
            ws1_msa_set[c2] = np.mean(cwise_ws1_msa)
            ws2_msa_set[c2] = np.mean(cwise_ws2_msa)
            ws1_sa50_set[c2] = np.mean(cwise_ws1_sa50)
            ws2_sa50_set[c2] = np.mean(cwise_ws2_sa50)

        fg_list.append(pd.DataFrame.from_dict([fg_set]))  # type: ignore
        bd_list.append(pd.DataFrame.from_dict([bd_set]))  # type: ignore
        ws1_msa_list.append(pd.DataFrame.from_dict([ws1_msa_set]))  # type: ignore
        ws2_msa_list.append(pd.DataFrame.from_dict([ws2_msa_set]))  # type: ignore
        ws1_sa50_list.append(pd.DataFrame.from_dict([ws1_sa50_set]))  # type: ignore
        ws2_sa50_list.append(pd.DataFrame.from_dict([ws2_sa50_set]))  # type: ignore

    f_df_fg = pd.concat(fg_list, ignore_index=True)
    f_df_bd = pd.concat(bd_list, ignore_index=True)
    f_df_ws1_msa = pd.concat(ws1_msa_list, ignore_index=True)
    f_df_ws2_msa = pd.concat(ws2_msa_list, ignore_index=True)
    f_df_ws1_sa50 = pd.concat(ws1_sa50_list, ignore_index=True)
    f_df_ws2_sa50 = pd.concat(ws2_sa50_list, ignore_index=True)

    tmp_csv_name = f"{source_choice}-sam" if sam_initialization else f"{source_choice}-scratch"
    csv_save_dir = f"./results/{tmp_csv_name}"
    os.makedirs(csv_save_dir, exist_ok=True)

    f_df_fg.to_csv(os.path.join(csv_save_dir, "foreground-dice.csv"))
    f_df_bd.to_csv(os.path.join(csv_save_dir, "boundary-dice.csv"))
    f_df_ws1_msa.to_csv(os.path.join(csv_save_dir, "watershed1-msa.csv"))
    f_df_ws2_msa.to_csv(os.path.join(csv_save_dir, "watershed2-msa.csv"))
    f_df_ws1_sa50.to_csv(os.path.join(csv_save_dir, "watershed1-sa50.csv"))
    f_df_ws2_sa50.to_csv(os.path.join(csv_save_dir, "watershed2-sa50.csv"))


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # overwrite to use complex device setups
    patch_shape = (512, 512)  # patch size used for training on livecell

    # get the desired loss function for training
    loss = common.get_loss_function(
        with_affinities=args.with_affinities  # takes care of calling the loss for training with affinities
    )

    # get the desired livecell loaders for training
    train_loader, val_loader, output_channels = common.get_my_livecell_loaders(
        args.input, patch_shape, args.cell_type,
        with_affinities=args.with_affinities  # this takes care of getting the loaders with affinities
    )

    # get the model for the training and inference on livecell dataset
    model = common.get_unetr_model(
        model_name=args.model_name, source_choice=args.source_choice, patch_shape=patch_shape,
        sam_initialization=args.do_sam_ini, output_channels=output_channels
    )
    model.to(device)

    # determining where to save the checkpoints and tensorboard logs
    save_root = os.path.join(
        args.save_root,
        "affinities" if args.with_affinities else "boundaries",
        f"{args.source_choice}-sam" if args.do_sam_ini else f"{args.source_choice}-scratch"
    ) if args.save_root is not None else args.save_root

    if args.train:
        print("2d UNETR training on LIVECell dataset")
        do_unetr_training(
            train_loader=train_loader, val_loader=val_loader, model=model, cell_types=args.cell_type,
            device=device, save_root=save_root, iterations=args.iterations, loss=loss
        )

    # determines the directory where the predictions will be saved
    root_save_dir = os.path.join(
        args.save_dir, f"unetr-{args.source_choice}-sam" if args.do_sam_ini else f"unetr-{args.source_choice}-scratch"
    )
    print("Predictions are saved in", root_save_dir)

    if args.predict:
        print("2d UNETR inference on LIVECell dataset")
        do_unetr_inference(
            input_path=args.input, device=device, model=model, cell_types=common.CELL_TYPES,
            root_save_dir=root_save_dir, save_root=save_root, with_affinities=args.with_affinities
        )

    if args.evaluate:
        print("2d UNETR evaluation on LIVECell dataset")
        do_unetr_evaluation(
            input_path=args.input, cell_types=common.CELL_TYPES, root_save_dir=root_save_dir,
            sam_initialization=args.do_sam_ini, source_choice=args.source_choice
        )


if __name__ == "__main__":
    parser = common.get_parser()
    args = parser.parse_args()
    main(args)
