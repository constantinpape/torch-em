import os
from glob import glob

import torch
import torch_em
import numpy as np
import pandas as pd
try:
    import imageio.v2 as imageio
except ImportError:
    import imageio

from elf.evaluation import dice_score
from torch_em.model import UNet2d
from torch_em.util.prediction import predict_with_padding
from tqdm import tqdm

from common import CELL_TYPES, get_parser, get_supervised_loader


def _train_cell_type(args, cell_type):
    model = UNet2d(in_channels=1, out_channels=1, initial_features=64, final_activation="Sigmoid")
    train_loader = get_supervised_loader(args, "train", cell_type)
    val_loader = get_supervised_loader(args, "val", cell_type)
    name = f"unet_source/{cell_type}"
    trainer = torch_em.default_segmentation_trainer(
        name=name,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=1e-4,
        mixed_precision=True,
        log_image_interval=100,
        save_root=args.save_root,
    )
    trainer.fit(iterations=args.n_iterations)


def run_training(args):
    for cell_type in args.cell_types:
        print("Start training for cell type:", cell_type)
        _train_cell_type(args, cell_type)


def check_loader(args, n_images=5):
    from torch_em.util.debug import check_loader

    cell_types = args.cell_types
    print("The cell types", cell_types, "were selected.")
    print("Checking the loader for the first cell type", cell_types[0])

    loader = get_supervised_loader(args)
    check_loader(loader, n_images)


def _eval_src(args, ct_src):
    ckpt = f"checkpoints/unet_source/{ct_src}"
    model = torch_em.util.get_trainer(ckpt).model

    image_folder = os.path.join(args.input, "images", "livecell_test_images")
    label_root = os.path.join(args.input, "annotations", "livecell_test_images")

    results = {"src": [ct_src]}
    device = torch.device("cuda")

    with torch.no_grad():
        for ct_trg in CELL_TYPES:
            label_paths = glob(os.path.join(label_root, ct_trg, "*.tif"))
            scores = []
            for label_path in tqdm(label_paths, desc=f"Predict for src={ct_src}, trgt={ct_trg}"):
                image_path = os.path.join(image_folder, os.path.basename(label_path))
                assert os.path.exists(image_path)
                image = imageio.imread(image_path)
                image = torch_em.transform.raw.standardize(image)
                pred = predict_with_padding(model, image, min_divisible=(16, 16), device=device).squeeze()
                labels = imageio.imread(label_path)
                assert image.shape == labels.shape
                score = dice_score(pred, labels, threshold_seg=None, threshold_gt=0)
                scores.append(score)
            results[ct_trg] = np.mean(scores)
    return pd.DataFrame(results)


def run_evaluation(args):
    results = []
    for ct in args.cell_types:
        res = _eval_src(args, ct)
        results.append(res)
    results = pd.concat(results)
    print("Evaluation results:")
    print(results)
    result_folder = "./results"
    os.makedirs(result_folder, exist_ok=True)
    results.to_csv(os.path.join(result_folder, "unet_source.csv"), index=False)


def main():
    parser = get_parser(default_iterations=50000)
    args = parser.parse_args()
    if args.phase in ("c", "check"):
        check_loader(args)
    elif args.phase in ("t", "train"):
        run_training(args)
    elif args.phase in ("e", "evaluate"):
        run_evaluation(args)
    else:
        raise ValueError(f"Got phase={args.phase}, expect one of check, train, evaluate.")


if __name__ == "__main__":
    main()
