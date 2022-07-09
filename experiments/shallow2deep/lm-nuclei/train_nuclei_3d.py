import os
from glob import glob

import torch_em
import torch_em.shallow2deep as shallow2deep
from torch_em.model import AnisotropicUNet


# TODO more publicly available ones? check initial experiments on this
DATA_ROOT = "/scratch/pape/s2d-lm"
DATASETS = ["root", "mouse-embryo"]


def normalize_datasets(datasets):
    wrong_ds = list(set(datasets) - set(DATASETS))
    if wrong_ds:
        raise ValueError(f"Unkown datasets: {wrong_ds}. Only {DATASETS} are supported")
    datasets = list(sorted(datasets))
    return datasets


# TODO require the data for this dataset
def require_ds(dataset):
    pass


def require_rfs_ds(dataset, n_rfs):
    out_folder = os.path.join("checkpoints/rfs", dataset)
    os.makedirs(out_folder, exist_ok=True)
    if len(glob(os.path.join(out_folder, "*.pkl"))) == n_rfs:
        return

    patch_shape_min = [24, 128, 128]
    patch_shape_max = [32, 256, 256]

    raw_transform = torch_em.transform.raw.normalize
    label_transform = shallow2deep.ForegroundTransform(ndim=3)

    paths, raw_key, label_key = require_ds(dataset)

    shallow2deep.prepare_shallow2deep_advanced(
        raw_paths=paths, raw_key=raw_key, label_paths=paths, label_key=label_key,
        patch_shape_min=patch_shape_min, patch_shape_max=patch_shape_max,
        n_forests=args.n_rfs, n_threads=args.n_threads,
        forests_per_stage=25, sample_fraction_per_stage=0.025,
        output_folder=out_folder, ndim=3,
        raw_transform=raw_transform, label_transform=label_transform,
        is_seg_dataset=True,
    )


def require_rfs(datasets, n_rfs):
    for ds in datasets:
        require_rfs_ds(ds, n_rfs)


# TODO
def get_loader(args, split, datasets):
    pass


def train_shallow2deep(args):
    datasets = normalize_datasets(args.datasets)
    name = f"s2d-lm-nuclei-{'_'.join(datasets)}"
    require_rfs(datasets, args.n_rfs)

    train_loader = get_loader(args, "train", datasets)
    val_loader = get_loader(args, "val", datasets)

    scale_factors = [[1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
    model = AnisotropicUNet(in_channels=1, out_channels=2, final_activation="Sigmoid",
                            scale_factors=scale_factors, initial_features=32)

    trainer = torch_em.default_segmentation_trainer(
        name, model, train_loader, val_loader, learning_rate=1.0e-4,
        device=args.device, log_image_interval=50
    )
    trainer.fit(args.n_iterations)


if __name__ == "__main__":
    parser = torch_em.util.parser_helper(require_input=False)
    parser.add_argument("--datasets", "-d", nargs="+", required=True)
    parser.add_argument("--n_rfs", type=int, default=200, help="Number of foersts per dataset")
    parser.add_argument("--n_threads", type=int, default=32)
    args = parser.parse_args()
    train_shallow2deep(args)
