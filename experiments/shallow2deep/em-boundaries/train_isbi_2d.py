import os
from glob import glob

import h5py
import numpy as np
import torch_em
import torch_em.shallow2deep as shallow2deep
from torch_em.model import UNet2d


def prepare_shallow2deep_isbi(args, out_folder):
    patch_shape_min = [1, 256, 256]
    patch_shape_max = [1, 512, 512]

    raw_transform = torch_em.transform.raw.normalize
    label_transform = shallow2deep.BoundaryTransform(ndim=2)

    shallow2deep.prepare_shallow2deep(
        raw_paths=args.input, raw_key="volumes/raw",
        label_paths=args.input, label_key="volumes/labels/neuron_ids_3d",
        patch_shape_min=patch_shape_min, patch_shape_max=patch_shape_max,
        n_forests=args.n_rfs, n_threads=args.n_threads,
        output_folder=out_folder, ndim=2,
        raw_transform=raw_transform, label_transform=label_transform,
    )


def get_isbi_loader(args, split, rf_folder):
    rf_paths = glob(os.path.join(rf_folder, "*.pkl"))
    rf_paths.sort()
    patch_shape = (1, 512, 512)
    with h5py.File(args.input, "r") as f:
        nz = f["volumes/raw"].shape[0]
    if split == "train":
        n_samples = 100
        roi = np.s_[:nz-2, :, :]
    elif split == "val":
        n_samples = 5
        roi = np.s_[nz-2:, :, :]
    else:
        raise ValueError(f"Wrong split: {split}")
    raw_transform = torch_em.transform.raw.normalize
    label_transform = torch_em.transform.BoundaryTransform(ndim=2)
    loader = shallow2deep.get_shallow2deep_loader(
        raw_paths=args.input, raw_key="volumes/raw",
        label_paths=args.input, label_key="volumes/labels/neuron_ids_3d",
        rf_paths=rf_paths,
        batch_size=args.batch_size, patch_shape=patch_shape, rois=roi,
        raw_transform=raw_transform, label_transform=label_transform,
        n_samples=n_samples, ndim=2, is_seg_dataset=True, shuffle=True,
        num_workers=8
    )
    return loader


def train_shallow2deep(args):
    name = "isbi2d"

    # check if we need to train the rfs for preparation
    rf_folder = os.path.join("checkpoints", name, "rfs")
    have_rfs = len(glob(os.path.join(rf_folder, "*.pkl"))) == args.n_rfs
    if not have_rfs:
        prepare_shallow2deep_isbi(args, rf_folder)
    assert os.path.exists(rf_folder)

    model = UNet2d(in_channels=1, out_channels=1, final_activation="Sigmoid")

    train_loader = get_isbi_loader(args, "train", rf_folder)
    val_loader = get_isbi_loader(args, "val", rf_folder)

    dice_loss = torch_em.loss.DiceLoss()
    trainer = torch_em.default_segmentation_trainer(
        name, model, train_loader, val_loader, loss=dice_loss, metric=dice_loss, learning_rate=1.0e-4,
        device=args.device, log_image_interval=50
    )
    trainer.fit(args.n_iterations)


def get_pseudolabel_loader(args, split, ckpt_name):
    patch_shape = (1, 512, 512)
    with h5py.File(args.input, "r") as f:
        nz = f["volumes/raw"].shape[0]
    if split == "train":
        n_samples = 100
        roi = np.s_[:nz-2, :, :]
    elif split == "val":
        n_samples = 5
        roi = np.s_[nz-2:, :, :]
    else:
        raise ValueError(f"Wrong split: {split}")
    ckpt = os.path.join("./checkpoints", ckpt_name)
    raw_transform = torch_em.transform.raw.normalize

    # tf trained on isbi
    rf_path = "./test_data/rf_0.pkl"
    ndim = 2
    filter_config = None

    loader = shallow2deep.get_pseudolabel_loader(
        raw_paths=args.input, raw_key="volumes/raw", checkpoint=ckpt,
        rf_config=(rf_path, ndim, filter_config),
        batch_size=args.batch_size, patch_shape=patch_shape, rois=roi,
        raw_transform=raw_transform, n_samples=n_samples,
        ndim=2, is_raw_dataset=True, shuffle=True, num_workers=0
    )
    return loader


def train_pseudo_label(args):
    name = "cremi2d-pseudo-label-isbi"

    model = UNet2d(in_channels=1, out_channels=1, final_activation="Sigmoid")

    train_loader = get_pseudolabel_loader(args, "train", "isbi2d")
    val_loader = get_pseudolabel_loader(args, "val", "isbi2d")

    dice_loss = torch_em.loss.DiceLoss()
    trainer = torch_em.default_segmentation_trainer(
        name, model, train_loader, val_loader, loss=dice_loss, metric=dice_loss, learning_rate=1.0e-4,
        device=args.device, log_image_interval=50
    )
    trainer.fit(args.n_iterations)


def check_loader(args, n=4):
    from torch_em.util.debug import check_loader
    loader = get_isbi_loader(args, "train", "./checkpoints/isbi2d/rfs")
    check_loader(loader, n)


if __name__ == "__main__":
    parser = torch_em.util.parser_helper()
    parser.add_argument("-p", "--pseudo_label", type=int, default=0)
    parser.add_argument("--n_rfs", type=int, default=500)
    parser.add_argument("--n_threads", type=int, default=32)
    args = parser.parse_args()
    if args.check:
        check_loader(args)
    elif args.pseudo_label:
        train_pseudo_label(args)
    else:
        train_shallow2deep(args)
