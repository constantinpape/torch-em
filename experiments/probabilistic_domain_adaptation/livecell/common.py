import argparse
import os
from glob import glob

try:
    import imageio.v2 as imageio
except ImportError:
    import imageio
import numpy as np
import pandas as pd
import torch
import torch_em

from elf.evaluation import dice_score
from torch_em.data.datasets.livecell import (get_livecell_loader,
                                             _download_livecell_images,
                                             _download_livecell_annotations)
from torch_em.model import UNet2d
from torch_em.util.prediction import predict_with_padding
from torchvision import transforms
from tqdm import tqdm

CELL_TYPES = ["A172", "BT474", "BV2", "Huh7", "MCF7", "SHSY5Y", "SkBr3", "SKOV3"]


#
# The augmentations we use for the LiveCELL experiments:
# - weak augmenations: blurring and additive gaussian noise
# - strong augmentations: TODO
#


def weak_augmentations(p=0.25):
    norm = torch_em.transform.raw.standardize
    aug = transforms.Compose([
        norm,
        transforms.RandomApply([torch_em.transform.raw.GaussianBlur()], p=p),
        transforms.RandomApply([torch_em.transform.raw.AdditiveGaussianNoise(
            scale=(0, 0.15), clip_kwargs=False)], p=p
        ),
    ])
    return torch_em.transform.raw.get_raw_transform(normalizer=norm, augmentation1=aug)


# TODO
def strong_augmentations():
    pass


#
# Model and prediction functionality: the models we use in all experiments
#

def get_unet():
    return UNet2d(in_channels=1, out_channels=1, initial_features=64, final_activation="Sigmoid", depth=4)


def load_model(model, ckpt, state="model_state", device=None):
    state = torch.load(os.path.join(ckpt, "best.pt"))[state]
    model.load_state_dict(state)
    if device is not None:
        model.to(device)
    return model


# use get_model and prediction_function to customize this, e.g. for using it with the PUNet
# set model_state to "teacher_state" when using this with a mean-teacher method
def evaluate_transfered_model(
    args, ct_src, method, get_model=get_unet, prediction_function=None, model_state="model_state"
):
    image_folder = os.path.join(args.input, "images", "livecell_test_images")
    label_root = os.path.join(args.input, "annotations", "livecell_test_images")

    results = {"src": [ct_src]}
    device = torch.device("cuda")

    thresh = args.confidence_threshold
    with torch.no_grad():
        for ct_trg in CELL_TYPES:

            if ct_trg == ct_src:
                results[ct_trg] = None
                continue

            out_folder = None if args.output is None else os.path.join(
                args.output, f"thresh-{thresh}", ct_src, ct_trg
            )
            if out_folder is not None:
                os.makedirs(out_folder, exist_ok=True)

            ckpt = f"checkpoints/{method}/thresh-{thresh}/{ct_src}/{ct_trg}"
            model = get_model()
            model = load_model(model, ckpt, device=device, state=model_state)

            label_paths = glob(os.path.join(label_root, ct_trg, "*.tif"))
            scores = []
            for label_path in tqdm(label_paths, desc=f"Predict for src={ct_src}, trgt={ct_trg}"):

                labels = imageio.imread(label_path)
                if out_folder is None:
                    out_path = None
                else:
                    out_path = os.path.join(out_folder, os.path.basename(label_path))
                    if os.path.exists(out_path):
                        pred = imageio.imread(out_path)
                        score = dice_score(pred, labels, threshold_seg=None, threshold_gt=0)
                        scores.append(score)
                        continue

                image_path = os.path.join(image_folder, os.path.basename(label_path))
                assert os.path.exists(image_path)
                image = imageio.imread(image_path)
                image = torch_em.transform.raw.standardize(image)
                pred = predict_with_padding(
                    model, image, min_divisible=(16, 16), device=device, prediction_function=prediction_function,
                ).squeeze()
                assert image.shape == labels.shape
                score = dice_score(pred, labels, threshold_seg=None, threshold_gt=0)
                if out_path is not None:
                    imageio.imwrite(out_path, pred)
                scores.append(score)

            results[ct_trg] = np.mean(scores)
    return pd.DataFrame(results)


# use get_model and prediction_function to customize this, e.g. for using it with the PUNet
def evaluate_source_model(args, ct_src, method, get_model=get_unet, prediction_function=None):
    if args.save_root is None:
        ckpt = f"checkpoints/{method}/{ct_src}"
    else:
        ckpt = args.save_root + f"checkpoints/{method}/{ct_src}"
    model = get_model()
    model = torch_em.util.get_trainer(ckpt).model

    image_folder = os.path.join(args.input, "images", "livecell_test_images")
    label_root = os.path.join(args.input, "annotations", "livecell_test_images")

    results = {"src": [ct_src]}
    device = torch.device("cuda")

    with torch.no_grad():
        for ct_trg in CELL_TYPES:

            out_folder = None if args.output is None else os.path.join(args.output, ct_src, ct_trg)
            if out_folder is not None:
                os.makedirs(out_folder, exist_ok=True)

            label_paths = glob(os.path.join(label_root, ct_trg, "*.tif"))
            scores = []
            for label_path in tqdm(label_paths, desc=f"Predict for src={ct_src}, trgt={ct_trg}"):

                labels = imageio.imread(label_path)
                if out_folder is None:
                    out_path = None
                else:
                    out_path = os.path.join(out_folder, os.path.basename(label_path))
                    if os.path.exists(out_path):
                        pred = imageio.imread(out_path)
                        score = dice_score(pred, labels, threshold_seg=None, threshold_gt=0)
                        scores.append(score)
                        continue

                image_path = os.path.join(image_folder, os.path.basename(label_path))
                assert os.path.exists(image_path)
                image = imageio.imread(image_path)
                image = torch_em.transform.raw.standardize(image)
                pred = predict_with_padding(
                    model, image, min_divisible=(16, 16), device=device, prediction_function=prediction_function
                ).squeeze()
                assert image.shape == labels.shape
                score = dice_score(pred, labels, threshold_seg=None, threshold_gt=0)
                if out_path is not None:
                    imageio.imwrite(out_path, pred)
                scores.append(score)

            results[ct_trg] = np.mean(scores)
    return pd.DataFrame(results)


#
# Other utility functions: loaders, parser
#


def _get_image_paths(args, split, cell_type):
    _download_livecell_images(args.input, download=True)
    image_paths, _ = _download_livecell_annotations(args.input, split, download=True,
                                                    cell_types=[cell_type], label_path=None)
    return image_paths


def get_unsupervised_loader(args, split, cell_type, teacher_augmentation, student_augmentation):
    patch_shape = (256, 256)

    def _parse_aug(aug):
        if aug == "weak":
            return weak_augmentations()
        elif aug == "strong":
            return strong_augmentations()
        assert callable(aug)
        return aug

    raw_transform = torch_em.transform.get_raw_transform()
    transform = torch_em.transform.get_augmentations(ndim=2)

    image_paths = _get_image_paths(args, split, cell_type)

    augmentations = (_parse_aug(teacher_augmentation), _parse_aug(student_augmentation))
    ds = torch_em.data.RawImageCollectionDataset(
        image_paths, patch_shape, raw_transform, transform,
        augmentations=augmentations
    )
    loader = torch_em.segmentation.get_data_loader(ds, batch_size=args.batch_size, num_workers=8, shuffle=True)
    return loader


def get_supervised_loader(args, split, cell_type, batch_size):
    patch_shape = (256, 256)
    loader = get_livecell_loader(
        args.input, patch_shape, split,
        download=True, binary=True, batch_size=batch_size,
        cell_types=[cell_type], num_workers=8, shuffle=True,
    )
    return loader


def get_parser(default_batch_size=8, default_iterations=int(1e5)):
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-p", "--phase", required=True)
    parser.add_argument("-b", "--batch_size", default=default_batch_size, type=int)
    parser.add_argument("-n", "--n_iterations", default=default_iterations, type=int)
    parser.add_argument("-s", "--save_root")
    parser.add_argument("-c", "--cell_types", nargs="+", default=CELL_TYPES)
    parser.add_argument("-o", "--output")
    return parser
