import os
import argparse
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from typing import Tuple, List

import imageio.v2 as imageio
from elf.evaluation import dice_score
from skimage.segmentation import find_boundaries

import torch
import torch_em
from torch_em.transform.raw import standardize
from torch_em.transform.label import labels_to_binary
from torch_em.data.datasets import get_livecell_loader
from torch_em.util.prediction import predict_with_halo


def get_unetr_model(
        model_name: str,
        source_choice: str,
        patch_shape: Tuple[int, int],
        sam_initialization: bool,
        output_channels: int
):
    """Returns the expected UNETR model
    """
    if source_choice == "torch-em":
        from torch_em import model as torch_em_models
        model = torch_em_models.UNETR(
            encoder=model_name, out_channels=output_channels,
            encoder_checkpoint_path="/scratch/usr/nimanwai/models/segment-anything/checkpoints/sam_vit_b_01ec64.pth" if sam_initialization else None
        )

    elif source_choice == "monai":
        from monai.networks import nets as monai_models
        model = monai_models.unetr.UNETR(
            in_channels=1,
            out_channels=output_channels,
            img_size=patch_shape,
            spatial_dims=2
        )
        model.out_channels = 2  # type: ignore

    else:
        raise ValueError("The available UNETR models are either from \"torch-em\" or \"monai\", choose from them")

    return model


def do_unetr_training(
        input_path: str,
        model,
        cell_types: List[str],
        patch_shape: Tuple[int, int],
        device: torch.device,
        save_root: str,
        iterations: int,
        sam_initialization: bool,
        source_choice: str
):
    train_loader = get_livecell_loader(
        path=input_path,
        split="train",
        patch_shape=patch_shape,
        batch_size=2,
        cell_types=[cell_types],
        download=True,
        boundaries=True,
        num_workers=8
    )

    val_loader = get_livecell_loader(
        path=input_path,
        split="val",
        patch_shape=patch_shape,
        batch_size=1,
        cell_types=[cell_types],
        download=True,
        boundaries=True,
        num_workers=8
    )

    _save_root = os.path.join(
        save_root,
        f"{source_choice}-sam" if sam_initialization else f"{source_choice}-scratch"
    ) if save_root is not None else save_root

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
        save_root=_save_root
    )

    trainer.fit(iterations)


def do_unetr_inference(
        input_path: str,
        device: torch.device,
        model,
        cell_types: List[str],
        root_save_dir: str,
        sam_initialization: bool,
        save_root: str,
        source_choice: str
):
    for ctype in cell_types:
        test_img_dir = os.path.join(input_path, "images", "livecell_test_images", "*")

        model_ckpt = os.path.join(save_root,
                                  f"{source_choice}-sam" if sam_initialization else f"{source_choice}-scratch",
                                  "checkpoints", f"livecell-{ctype}", "best.pt")
        assert os.path.exists(model_ckpt)

        model.load_state_dict(torch.load(model_ckpt, map_location=torch.device('cpu'))["model_state"])
        model.to(device)
        model.eval()

        with torch.no_grad():
            for img_path in glob(test_img_dir):
                fname = os.path.split(img_path)[-1]

                input_img = imageio.imread(img_path)
                input_img = standardize(input_img)
                outputs = predict_with_halo(input_img, model, gpu_ids=[device], block_shape=[384, 384], halo=[64, 64])

                fg, bd = outputs[0, :, :], outputs[1, :, :]

                fg_save_dir = os.path.join(root_save_dir, f"src-{ctype}", "foreground")
                bd_save_dir = os.path.join(root_save_dir, f"src-{ctype}", "boundary")

                os.makedirs(fg_save_dir, exist_ok=True)
                os.makedirs(bd_save_dir, exist_ok=True)

                imageio.imwrite(os.path.join(fg_save_dir, fname), fg)
                imageio.imwrite(os.path.join(bd_save_dir, fname), bd)


def do_unetr_evaluation(
        input_path: str,
        cell_types: List[str],
        root_save_dir: str,
        sam_initialization: bool,
        source_choice: str
):
    fg_list, bd_list = [], []

    for c1 in cell_types:
        _save_dir = os.path.join(root_save_dir, f"src-{c1}")

        fg_set, bd_set = {"CELL TYPE": c1}, {"CELL TYPE": c1}
        for c2 in tqdm(cell_types, desc=f"Evaluation on {c1} source models"):
            fg_dir = os.path.join(_save_dir, "foreground")
            bd_dir = os.path.join(_save_dir, "boundary")

            gt_dir = os.path.join(input_path, "annotations", "livecell_test_images", c2, "*")
            cwise_fg, cwise_bd = [], []
            for gt_path in glob(gt_dir):
                fname = os.path.split(gt_path)[-1]

                gt = imageio.imread(gt_path)
                fg = imageio.imread(os.path.join(fg_dir, fname))
                bd = imageio.imread(os.path.join(bd_dir, fname))

                true_fg = labels_to_binary(gt)
                true_bd = find_boundaries(gt)

                cwise_fg.append(dice_score(fg, true_fg, threshold_gt=0))
                cwise_bd.append(dice_score(bd, true_bd, threshold_gt=0))

            fg_set[c2] = np.mean(cwise_fg)
            bd_set[c2] = np.mean(cwise_bd)

        fg_list.append(pd.DataFrame.from_dict([fg_set]))  # type: ignore
        bd_list.append(pd.DataFrame.from_dict([bd_set]))  # type: ignore

    f_df_fg = pd.concat(fg_list, ignore_index=True)
    f_df_bd = pd.concat(bd_list, ignore_index=True)

    csv_save_dir = "./results/"
    os.makedirs(csv_save_dir, exist_ok=True)

    tmp_csv_name = f"{source_choice}-sam" if sam_initialization else f"{source_choice}-scratch"
    f_df_fg.to_csv(os.path.join(csv_save_dir, f"foreground-unetr-{tmp_csv_name}-results.csv"))
    f_df_bd.to_csv(os.path.join(csv_save_dir, f"boundary-unetr-{tmp_csv_name}-results.csv"))


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patch_shape = (512, 512)
    output_channels = 2

    all_cell_types = ["A172", "BT474", "BV2", "Huh7", "MCF7", "SHSY5Y", "SkBr3", "SKOV3"]

    model = get_unetr_model(
        model_name=args.model_name,
        source_choice=args.source_choice,
        patch_shape=patch_shape,
        sam_initialization=args.do_sam_ini,
        output_channels=output_channels
    )
    model.to(device)

    if args.train:
        print("2d UNETR training on LiveCell dataset")
        do_unetr_training(
            input_path=args.input,
            model=model,
            cell_types=args.cell_type,
            patch_shape=patch_shape,
            device=device,
            save_root=args.save_root,
            iterations=args.iterations,
            sam_initialization=args.do_sam_ini,
            source_choice=args.source_choice
        )

    root_save_dir = os.path.join(
        args.save_dir,
        f"unetr-torch-em-sam-{args.model_name}" if args.do_sam_ini else f"unetr-torch-em-scratch-{args.model_name}"
    )

    if args.predict:
        print("2d UNETR inference on LiveCell dataset")
        do_unetr_inference(
            input_path=args.input,
            device=device,
            model=model,
            cell_types=all_cell_types,
            root_save_dir=root_save_dir,
            sam_initialization=args.do_sam_ini,
            save_root=args.save_root,
            source_choice=args.source_choice
        )
    if args.evaluate:
        print("2d UNETR evaluation on LiveCell dataset")
        do_unetr_evaluation(
            input_path=args.input,
            cell_types=all_cell_types,
            root_save_dir=root_save_dir,
            sam_initialization=args.do_sam_ini,
            source_choice=args.source_choice
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action='store_true',
                        help="Enables UNETR training on LiveCell dataset")

    parser.add_argument("--predict", action='store_true',
                        help="Enables UNETR prediction on LiveCell dataset")

    parser.add_argument("--evaluate", action='store_true',
                        help="Enables UNETR evaluation on LiveCell dataset")

    parser.add_argument("--source_choice", type=str, default="torch-em",
                        help="The source where the model comes from, i.e. either torch-em / monai")

    parser.add_argument("-m", "--model_name", type=str, default="vit_b",
                        help="Name of the ViT to use as the encoder in UNETR")

    parser.add_argument("--do_sam_ini", action='store_true',
                        help="Enables initializing UNETR with SAM's ViT weights")

    parser.add_argument("-c", "--cell_type", type=str, default=None,
                        help="Choice of cell-type for doing the training")

    parser.add_argument("-i", "--input", type=str, default="/scratch/usr/nimanwai/data/livecell",
                        help="Path where the dataset already exists/will be downloaded by the dataloader")

    parser.add_argument("-s", "--save_root", type=str, default="/scratch/usr/nimanwai/models/unetr/",
                        help="Path where checkpoints and logs will be saved")

    parser.add_argument("--save_dir", type=str, default="/scratch/usr/nimanwai/predictions/unetr",
                        help="Path to save predictions from UNETR model")

    parser.add_argument("--iterations", type=int, default=10000)

    args = parser.parse_args()
    main(args)
