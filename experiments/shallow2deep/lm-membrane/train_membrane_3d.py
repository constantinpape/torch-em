import os
from glob import glob

import torch
import torch_em
import torch_em.shallow2deep as shallow2deep
from elf.io import open_file
from torch_em.model import UNet3d
from torch_em.data.datasets.mouse_embryo import _require_embryo_data
from torch_em.data.datasets.plantseg import _require_plantseg_data
from torch_em.data.datasets.pnas_arabidopsis import _require_pnas_data


# any more publicly available datasets?
DATA_ROOT = "/scratch/pape/s2d-lm-boundaries"
DATASETS = ["mouse-embryo", "ovules", "pnas", "root"]


def normalize_datasets(datasets):
    wrong_ds = list(set(datasets) - set(DATASETS))
    if wrong_ds:
        raise ValueError(f"Unkown datasets: {wrong_ds}. Only {DATASETS} are supported")
    datasets = list(sorted(datasets))
    return datasets


def _get_pnas_paths(pnas_root, split):
    folders = glob(os.path.join(pnas_root, "plant*"))
    folders.sort()
    paths = []
    for folder in folders:
        all_paths = glob(os.path.join(folder, "*.h5"))
        all_paths.sort()
        this_paths = [
            pp for pp in all_paths if "labels/membrane" in open_file(pp, "r")
        ]
        paths.extend(
            this_paths[:-2] if split == "train" else this_paths[-2:]
        )
    return paths


def require_ds(dataset):
    os.makedirs(DATA_ROOT, exist_ok=True)
    data_path = os.path.join(DATA_ROOT, dataset)
    if dataset == "root":
        _require_plantseg_data(data_path, True, "root", "train")
        paths = glob(os.path.join(data_path, "root_train", "*.h5"))
        raw_key, label_key = "raw", "label"
    elif dataset == "ovules":
        _require_plantseg_data(data_path, True, "ovules", "train")
        paths = glob(os.path.join(data_path, "ovules_train", "*.h5"))
        raw_key, label_key = "raw", "label"
    elif dataset == "pnas":
        _require_pnas_data(data_path, True)
        paths = _get_pnas_paths(data_path, split="train")
        raw_key, label_key = "raw/membrane", "labels/membrane"
    elif dataset == "mouse-embryo":
        _require_embryo_data(data_path, True)
        paths = glob(os.path.join(data_path, "Membrane", "train", "*.h5"))
        raw_key, label_key = "raw", "label"
    return paths, raw_key, label_key


def require_rfs_ds(dataset, n_rfs, sampling_strategy):
    out_folder = os.path.join(DATA_ROOT, f"rfs3d-{sampling_strategy}", dataset)
    os.makedirs(out_folder, exist_ok=True)
    if len(glob(os.path.join(out_folder, "*.pkl"))) == n_rfs:
        return

    patch_shape_min = [24, 96, 96]
    patch_shape_max = [32, 128, 128]

    raw_transform = torch_em.transform.raw.normalize
    label_transform = shallow2deep.BoundaryTransform(ndim=3)

    paths, raw_key, label_key = require_ds(dataset)

    sampler = torch_em.data.MinForegroundSampler(min_fraction=0.25, background_id=1 if dataset == "root" else 0)
    shallow2deep.prepare_shallow2deep_advanced(
        raw_paths=paths, raw_key=raw_key, label_paths=paths, label_key=label_key,
        patch_shape_min=patch_shape_min, patch_shape_max=patch_shape_max,
        n_forests=args.n_rfs, n_threads=args.n_threads,
        forests_per_stage=25, sample_fraction_per_stage=0.025,
        output_folder=out_folder, ndim=3,
        raw_transform=raw_transform, label_transform=label_transform,
        is_seg_dataset=True, sampler=sampler, sampling_strategy=sampling_strategy,
        sampling_kwargs=dict(tile_shape=[25, 25, 25]),
    )


def require_rfs(datasets, n_rfs, sampling_strategy):
    for ds in datasets:
        require_rfs_ds(ds, n_rfs, sampling_strategy)


def get_ds(file_pattern, rf_pattern, n_samples, sampler):
    raw_transform = torch_em.transform.raw.normalize
    label_transform = torch_em.transform.BoundaryTransform(ndim=3, add_binary_target=False)
    patch_shape = [64, 128, 128]
    raw_key, label_key = "raw", "label"
    if isinstance(file_pattern, str):
        paths = glob(file_pattern)
        paths.sort()
    else:
        paths = file_pattern
    assert len(paths) > 0
    rf_paths = glob(rf_pattern)
    rf_paths.sort()
    assert len(rf_paths) > 0
    return shallow2deep.shallow2deep_dataset.get_shallow2deep_dataset(
        paths, raw_key, paths, label_key, rf_paths,
        patch_shape=patch_shape, label_transform=label_transform,
        raw_transform=raw_transform,
        n_samples=n_samples, sampler=sampler,
    )


def get_loader(args, split, dataset_names):
    datasets = []
    n_samples = 500 if split == "train" else 25
    if "mouse-embryo" in dataset_names:
        ds_name = "mouse-embryo"
        file_pattern = os.path.join(DATA_ROOT, ds_name, "Membrane", split, "*.h5")
        rf_pattern = os.path.join(DATA_ROOT, f"rfs3d-{args.sampling_strategy}", ds_name, "*.pkl")
        sampler = torch_em.data.MinForegroundSampler(min_fraction=0.25, background_id=0)
        datasets.append(get_ds(file_pattern, rf_pattern, n_samples, sampler))
    if "ovules" in dataset_names:
        ds_name = "ovules"
        _require_plantseg_data(os.path.join(DATA_ROOT, ds_name), True, ds_name, split)
        file_pattern = os.path.join(DATA_ROOT, ds_name, f"ovules_{split}", "*.h5")
        rf_pattern = os.path.join(DATA_ROOT, f"rfs3d-{args.sampling_strategy}", ds_name, "*.pkl")
        sampler = torch_em.data.MinForegroundSampler(min_fraction=0.25, background_id=0)
        datasets.append(get_ds(file_pattern, rf_pattern, n_samples, sampler))
    if "pnas" in dataset_names:
        ds_name = "pnas"
        files = _get_pnas_paths(os.path.join(DATA_ROOT, ds_name), split=split)
        rf_pattern = os.path.join(DATA_ROOT, f"rfs2d-{args.sampling_strategy}", ds_name, "*.pkl")
        sampler = torch_em.data.MinForegroundSampler(min_fraction=0.4, background_id=0)
        datasets.append(get_ds(files, rf_pattern, n_samples, sampler=sampler,
                               raw_key="raw/membrane", label_key="labels/membrane"))
    if "root" in dataset_names:
        ds_name = "root"
        _require_plantseg_data(os.path.join(DATA_ROOT, ds_name), True, ds_name, split)
        file_pattern = os.path.join(DATA_ROOT, ds_name, f"root_{split}", "*.h5")
        rf_pattern = os.path.join(DATA_ROOT, f"rfs3d-{args.sampling_strategy}", ds_name, "*.pkl")
        sampler = torch_em.data.MinForegroundSampler(min_fraction=0.25, background_id=0)
        datasets.append(get_ds(file_pattern, rf_pattern, n_samples, sampler=sampler))
    ds = torch_em.data.concat_dataset.ConcatDataset(*datasets)
    loader = torch.utils.data.DataLoader(
        ds, shuffle=True, batch_size=args.batch_size, num_workers=12
    )
    loader.shuffle = True
    return loader


def train_shallow2deep(args):
    datasets = normalize_datasets(args.datasets)
    name = f"s2d-lm-membrane-{'_'.join(datasets)}-3d-{args.sampling_strategy}"
    require_rfs(datasets, args.n_rfs, args.sampling_strategy)

    train_loader = get_loader(args, "train", datasets)
    val_loader = get_loader(args, "val", datasets)

    model = UNet3d(in_channels=1, out_channels=2, final_activation="Sigmoid", depth=4, initial_features=32)

    trainer = torch_em.default_segmentation_trainer(
        name, model, train_loader, val_loader, learning_rate=1.0e-4,
        device=args.device, log_image_interval=50
    )
    trainer.fit(args.n_iterations)


if __name__ == "__main__":
    parser = torch_em.util.parser_helper(require_input=False)
    parser.add_argument("--datasets", "-d", nargs="+", default=DATASETS)
    parser.add_argument("--n_rfs", type=int, default=500, help="Number of forests per dataset")
    parser.add_argument("--n_threads", type=int, default=32)
    parser.add_argument("--sampling_strategy", "-s", default="worst_tiles")
    args = parser.parse_args()
    train_shallow2deep(args)
