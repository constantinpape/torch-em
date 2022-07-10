import os
from glob import glob

import numpy as np
import torch
import torch_em
import torch_em.shallow2deep as shallow2deep
from torch_em.model import UNet2d
from torch_em.data.datasets.covid_if import _download_covid_if
from torch_em.data.datasets.mouse_embryo import _require_embryo_data
from torch_em.data.datasets.plantseg import _require_plantseg_data


# any more publicly available datasets?
DATA_ROOT = "/scratch/pape/s2d-lm-boundaries"
DATASETS = ["covid-if", "livecell", "mouse-embryo", "ovules", "root"]


def normalize_datasets(datasets):
    wrong_ds = list(set(datasets) - set(DATASETS))
    if wrong_ds:
        raise ValueError(f"Unkown datasets: {wrong_ds}. Only {DATASETS} are supported")
    datasets = list(sorted(datasets))
    return datasets


# def _require_livecell_data(data_path, split):
#     _download_livecell_images(data_path, True)
#     image_paths, label_paths = _download_livecell_annotations(daa_path, split, True)
#     return image_paths, label_paths


def require_ds(dataset):
    os.makedirs(DATA_ROOT, exist_ok=True)
    data_path = os.path.join(DATA_ROOT, dataset)
    if dataset == "covid-if":
        _download_covid_if(data_path, True)
        paths = glob(os.path.join(data_path, "*.h5"))
        paths.sort()
        paths = paths[:-5]
        raw_key, label_key = "raw/serum_IgG/s0", "labels/cells/s0"
    elif dataset == "livecell":
        raise NotImplementedError
    elif dataset == "mouse-embryo":
        _require_embryo_data(data_path, True)
        paths = glob(os.path.join(data_path, "Membrane", "train", "*.h5"))
        raw_key, label_key = "raw", "label"
    elif dataset == "ovules":
        _require_plantseg_data(data_path, True, "ovules", "train")
        paths = glob(os.path.join(data_path, "ovules_train", "*.h5"))
        raw_key, label_key = "raw", "label"
    elif dataset == "root":
        _require_plantseg_data(data_path, True, "root", "train")
        paths = glob(os.path.join(data_path, "root_train", "*.h5"))
        raw_key, label_key = "raw", "label"
    return paths,  raw_key, label_key


def require_rfs_ds(dataset, n_rfs):
    out_folder = os.path.join(DATA_ROOT, "rfs2d", dataset)
    os.makedirs(out_folder, exist_ok=True)
    if len(glob(os.path.join(out_folder, "*.pkl"))) == n_rfs:
        return

    # genuine 2d datasets
    if dataset in ("covid-if", "livecell"):
        patch_shape_min = [224, 224]
        patch_shape_max = [256, 256]
    else:  # 3d datasets we use for 2d training
        patch_shape_min = [1, 224, 224]
        patch_shape_max = [1, 256, 256]

    raw_transform = torch_em.transform.raw.normalize
    label_transform = shallow2deep.BoundaryTransform(ndim=2)

    paths, raw_key, label_key = require_ds(dataset)

    sampler = torch_em.data.MinForegroundSampler(min_fraction=0.05, background_id=1 if dataset == "root" else 0)
    shallow2deep.prepare_shallow2deep_advanced(
        raw_paths=paths, raw_key=raw_key, label_paths=paths, label_key=label_key,
        patch_shape_min=patch_shape_min, patch_shape_max=patch_shape_max,
        n_forests=args.n_rfs, n_threads=args.n_threads,
        forests_per_stage=25, sample_fraction_per_stage=0.10,
        output_folder=out_folder, ndim=2,
        raw_transform=raw_transform, label_transform=label_transform,
        is_seg_dataset=True, sampler=sampler
    )


def require_rfs(datasets, n_rfs):
    for ds in datasets:
        require_rfs_ds(ds, n_rfs)


def get_ds(file_pattern, rf_pattern, n_samples, is3d_data=True, path_selection=None, raw_key="raw", label_key="label"):
    label_transform = torch_em.transform.BoundaryTransform(ndim=2, add_binary_target=False)
    patch_shape = [1, 256, 256] if is3d_data else [256, 256]
    paths = glob(file_pattern)
    paths.sort()
    assert len(paths) > 0
    if path_selection is not None:
        paths = paths[path_selection]
    rf_paths = glob(rf_pattern)
    rf_paths.sort()
    assert len(rf_paths) > 0
    return shallow2deep.shallow2deep_dataset.get_shallow2deep_dataset(
        paths, raw_key, paths, label_key, rf_paths,
        patch_shape=patch_shape, label_transform=label_transform,
        n_samples=n_samples, ndim=2
    )


def get_loader(args, split, dataset_names):
    datasets = []
    n_samples = 500 if split == "train" else 25
    if "covid-if" in dataset_names:
        ds_name = "covid-if"
        file_pattern = os.path.join(DATA_ROOT, ds_name, "*.h5")
        rf_pattern = os.path.join(DATA_ROOT, "rfs2d", ds_name, "*.pkl")
        raw_key, label_key = "raw/serum_IgG/s0", "labels/cells/s0"
        datasets.append(get_ds(
            file_pattern, rf_pattern, n_samples, is3d_data=False, raw_key=raw_key, label_key=label_key,
            path_selection=np.s_[:-5] if split == "train" else np.s_[-5:]
        ))
    if "livecell" in dataset_names:
        raise NotImplementedError
    if "mouse-embryo" in dataset_names:
        ds_name = "mouse-embryo"
        file_pattern = os.path.join(DATA_ROOT, ds_name, "Membrane", split, "*.h5")
        rf_pattern = os.path.join(DATA_ROOT, "rfs2d", ds_name, "*.pkl")
        datasets.append(get_ds(file_pattern, rf_pattern, n_samples))
    if "ovules" in dataset_names:
        ds_name = "ovules"
        _require_plantseg_data(os.path.join(DATA_ROOT, ds_name), True, ds_name, split)
        file_pattern = os.path.join(DATA_ROOT, ds_name, f"ovules_{split}", "*.h5")
        rf_pattern = os.path.join(DATA_ROOT, "rfs2d", ds_name, "*.pkl")
        datasets.append(get_ds(file_pattern, rf_pattern, n_samples))
    if "root" in dataset_names:
        ds_name = "root"
        _require_plantseg_data(os.path.join(DATA_ROOT, ds_name), True, ds_name, split)
        file_pattern = os.path.join(DATA_ROOT, ds_name, f"root_{split}", "*.h5")
        rf_pattern = os.path.join(DATA_ROOT, "rfs2d", ds_name, "*.pkl")
        datasets.append(get_ds(file_pattern, rf_pattern, n_samples))
    ds = torch_em.data.concat_dataset.ConcatDataset(*datasets)
    loader = torch.utils.data.DataLoader(
        ds, shuffle=True, batch_size=args.batch_size, num_workers=12
    )
    loader.shuffle = True
    return loader


def train_shallow2deep(args):
    datasets = normalize_datasets(args.datasets)
    name = f"s2d-lm-membrane-{'_'.join(datasets)}-2d"
    require_rfs(datasets, args.n_rfs)

    train_loader = get_loader(args, "train", datasets)
    val_loader = get_loader(args, "val", datasets)
    model = UNet2d(in_channels=1, out_channels=1, final_activation="Sigmoid",
                   depth=4, initial_features=64)

    trainer = torch_em.default_segmentation_trainer(
        name, model, train_loader, val_loader, learning_rate=1.0e-4,
        device=args.device, log_image_interval=50
    )
    trainer.fit(args.n_iterations)


if __name__ == "__main__":
    parser = torch_em.util.parser_helper(require_input=False, default_batch_size=4)
    parser.add_argument("--datasets", "-d", nargs="+", default=DATASETS)
    parser.add_argument("--n_rfs", type=int, default=150, help="Number of foersts per dataset")
    parser.add_argument("--n_threads", type=int, default=32)
    args = parser.parse_args()
    train_shallow2deep(args)
