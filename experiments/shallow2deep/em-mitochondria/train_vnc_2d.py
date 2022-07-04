import os
from glob import glob

import numpy as np
import torch_em
import torch_em.shallow2deep as shallow2deep
from torch_em.model import UNet2d
from torch_em.data.datasets.vnc import _get_vnc_data


def prepare_shallow2deep(args, out_folder):
    patch_shape_min = [1, 256, 256]
    patch_shape_max = [1, 512, 512]

    raw_transform = torch_em.transform.raw.normalize
    label_transform = shallow2deep.ForegroundTransform(ndim=2)

    path = os.path.join(args.input, "vnc_train.h5")
    raw_key = "raw"
    label_key = "labels/mitochondria"

    if args.train_advanced:
        shallow2deep.prepare_shallow2deep_advanced(
            raw_paths=path, raw_key=raw_key, label_paths=path, label_key=label_key,
            patch_shape_min=patch_shape_min, patch_shape_max=patch_shape_max,
            n_forests=args.n_rfs, n_threads=args.n_threads,
            forests_per_stage=25, sample_fraction_per_stage=0.05,
            output_folder=out_folder, ndim=2,
            raw_transform=raw_transform, label_transform=label_transform,
            is_seg_dataset=True,
        )
    else:
        shallow2deep.prepare_shallow2deep(
            raw_paths=path, raw_key=raw_key, label_paths=path, label_key=label_key,
            patch_shape_min=patch_shape_min, patch_shape_max=patch_shape_max,
            n_forests=args.n_rfs, n_threads=args.n_threads,
            output_folder=out_folder, ndim=2,
            raw_transform=raw_transform, label_transform=label_transform,
            is_seg_dataset=True,
        )


def get_loader(args, split, rf_folder):
    rf_paths = glob(os.path.join(rf_folder, "*.pkl"))
    rf_paths.sort()
    patch_shape = (1, 512, 512)

    path = os.path.join(args.input, "vnc_train.h5")
    roi = np.s_[:18, :, :] if split == "train" else np.s_[18:, :, :]
    n_samples = 500 if split == "train" else 25

    raw_transform = torch_em.transform.raw.normalize
    label_transform = torch_em.transform.BoundaryTransform(ndim=2, add_binary_target=True)
    loader = shallow2deep.get_shallow2deep_loader(
        raw_paths=path, raw_key="raw",
        label_paths=path, label_key="labels/mitochondria",
        rf_paths=rf_paths,
        batch_size=args.batch_size, patch_shape=patch_shape,
        raw_transform=raw_transform, label_transform=label_transform,
        n_samples=n_samples, ndim=2, is_seg_dataset=True, shuffle=True,
        num_workers=12, rois=roi
    )
    return loader


def train_shallow2deep(args):
    name = "shallow2deep-em-mitochondria"
    if args.train_advanced:
        name += "-advanced"
    _get_vnc_data(args.input, download=True)

    # check if we need to train the rfs for preparation
    rf_folder = os.path.join("checkpoints", name, "rfs")
    have_rfs = len(glob(os.path.join(rf_folder, "*.pkl"))) == args.n_rfs
    if not have_rfs:
        prepare_shallow2deep(args, rf_folder)
    assert os.path.exists(rf_folder)

    model = UNet2d(in_channels=1, out_channels=2, final_activation="Sigmoid",
                   depth=4, initial_features=64)

    train_loader = get_loader(args, "train", rf_folder)
    val_loader = get_loader(args, "val", rf_folder)

    trainer = torch_em.default_segmentation_trainer(
        name, model, train_loader, val_loader, learning_rate=1.0e-4,
        device=args.device, log_image_interval=50
    )
    trainer.fit(args.n_iterations)


if __name__ == "__main__":
    parser = torch_em.util.parser_helper()
    parser.add_argument("--train_advanced", "-a", type=int, default=0)
    parser.add_argument("--n_rfs", type=int, default=500)
    parser.add_argument("--n_threads", type=int, default=32)
    args = parser.parse_args()
    train_shallow2deep(args)
