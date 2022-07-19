import os
from glob import glob

import torch
import torch_em
import torch_em.shallow2deep as shallow2deep
from torch_em.data.datasets.mitoem import _require_mitoem_sample
from torch_em.data.datasets.vnc import _get_vnc_data
from torch_em.model import UNet2d


DATA_ROOT = "/scratch/pape/s2d-mitochondria"
DATASETS = ["mitoem", "vnc"]


def normalize_datasets(datasets):
    wrong_ds = list(set(datasets) - set(DATASETS))
    if wrong_ds:
        raise ValueError(f"Unkown datasets: {wrong_ds}. Only {DATASETS} are supported")
    datasets = list(sorted(datasets))
    return datasets


def require_ds(dataset):
    os.makedirs(DATA_ROOT, exist_ok=True)
    data_path = os.path.join(DATA_ROOT, dataset)
    if dataset == "mitoem":
        if not os.path.exists(data_path):
            _require_mitoem_sample(data_path, sample="human", download=True)
            _require_mitoem_sample(data_path, sample="rat", download=True)
        paths = [
            os.path.join(data_path, "human_train.n5"),
            os.path.join(data_path, "rat_train.n5"),
        ]
        assert all(os.path.exists(pp) for pp in paths)
        raw_key, label_key = "raw", "labels"
    elif dataset == "vnc":
        _get_vnc_data(data_path, True)
        paths = [os.path.join(data_path, "vnc_train.h5")]
        raw_key, label_key = "raw", "labels/mitochondria"
    return paths, raw_key, label_key


def require_rfs_ds(dataset, n_rfs, sampling_strategy):
    if sampling_strategy is None:
        out_folder = os.path.join(DATA_ROOT, "rfs2d", dataset)
    else:
        out_folder = os.path.join(DATA_ROOT, f"rfs2d-{sampling_strategy}", dataset)
    os.makedirs(out_folder, exist_ok=True)
    if len(glob(os.path.join(out_folder, "*.pkl"))) == n_rfs:
        return

    patch_shape_min = [1, 128, 128]
    patch_shape_max = [1, 256, 256]

    raw_transform = torch_em.transform.raw.normalize
    label_transform = shallow2deep.ForegroundTransform(ndim=2)

    paths, raw_key, label_key = require_ds(dataset)

    if sampling_strategy == "vanilla":
        shallow2deep.prepare_shallow2deep(
            raw_paths=paths, raw_key=raw_key, label_paths=paths, label_key=label_key,
            patch_shape_min=patch_shape_min, patch_shape_max=patch_shape_max,
            n_forests=args.n_rfs, n_threads=args.n_threads,
            output_folder=out_folder, ndim=2,
            raw_transform=raw_transform, label_transform=label_transform,
            is_seg_dataset=True,
        )
    else:
        sampling_strategy = "worst_points" if sampling_strategy is None else sampling_strategy
        shallow2deep.prepare_shallow2deep_advanced(
            raw_paths=paths, raw_key=raw_key, label_paths=paths, label_key=label_key,
            patch_shape_min=patch_shape_min, patch_shape_max=patch_shape_max,
            n_forests=args.n_rfs, n_threads=args.n_threads,
            forests_per_stage=25, sample_fraction_per_stage=0.10,
            output_folder=out_folder, ndim=2,
            raw_transform=raw_transform, label_transform=label_transform,
            is_seg_dataset=True, sampling_strategy=sampling_strategy,
        )


def require_rfs(datasets, n_rfs, sampling_strategy):
    for ds in datasets:
        require_rfs_ds(ds, n_rfs, sampling_strategy)


def get_ds(file_pattern, rf_pattern, n_samples, label_key):
    label_transform = torch_em.transform.BoundaryTransform(ndim=2, add_binary_target=True)
    patch_shape = (1, 512, 512)
    paths = glob(file_pattern)
    paths.sort()
    assert len(paths) > 0
    rf_paths = glob(rf_pattern)
    rf_paths.sort()
    assert len(rf_paths) > 0
    raw_key = "raw"
    return shallow2deep.shallow2deep_dataset.get_shallow2deep_dataset(
        paths, raw_key, paths, label_key, rf_paths,
        patch_shape=patch_shape, label_transform=label_transform,
        n_samples=n_samples, ndim=2,
    )


def get_loader(args, split, dataset_names):
    datasets = []
    n_samples = 500 if split == "train" else 25
    if "mitoem" in dataset_names:
        ds_name = "mitoem"
        file_pattern = os.path.join(DATA_ROOT, ds_name, f"*_{split}.n5")
        rf_pattern = os.path.join(DATA_ROOT, "rfs2d", ds_name, "*.pkl")
        datasets.append(get_ds(file_pattern, rf_pattern, n_samples, label_key="labels"))
    if "vnc" in dataset_names and split == "train":
        ds_name = "vnc"
        file_pattern = os.path.join(DATA_ROOT, ds_name, f"vnc_{split}.h5")
        rf_pattern = os.path.join(DATA_ROOT, "rfs2d", ds_name, "*.pkl")
        datasets.append(get_ds(file_pattern, rf_pattern, n_samples, label_key="labels/mitochondria"))
    ds = torch_em.data.concat_dataset.ConcatDataset(*datasets) if len(datasets) > 1 else datasets[0]
    loader = torch.utils.data.DataLoader(
        ds, shuffle=True, batch_size=args.batch_size, num_workers=12
    )
    loader.shuffle = True
    return loader


def train_shallow2deep(args):
    datasets = normalize_datasets(args.datasets)
    name = f"s2d-em-mitos-{'_'.join(datasets)}-2d"
    if args.sampling_strategy is not None:
        name += f"-{args.sampling_strategy}"
    require_rfs(datasets, args.n_rfs, args.sampling_strategy)

    model = UNet2d(in_channels=1, out_channels=2, final_activation="Sigmoid",
                   depth=4, initial_features=64)

    train_loader = get_loader(args, "train", datasets)
    val_loader = get_loader(args, "val", datasets)

    trainer = torch_em.default_segmentation_trainer(
        name, model, train_loader, val_loader, learning_rate=1.0e-4,
        device=args.device, log_image_interval=50
    )
    trainer.fit(args.n_iterations)


def check(args, train=True, val=True, n_images=2):
    from torch_em.util.debug import check_loader
    datasets = normalize_datasets(args.datasets)
    if train:
        print("Check train loader")
        loader = get_loader(args, "train", datasets)
        check_loader(loader, n_images)
    if val:
        print("Check val loader")
        loader = get_loader(args, "val", datasets)
        check_loader(loader, n_images)


if __name__ == "__main__":
    parser = torch_em.util.parser_helper(require_input=False, default_batch_size=4)
    parser.add_argument("--datasets", "-d", nargs="+", default=DATASETS)
    parser.add_argument("--n_rfs", type=int, default=500)
    parser.add_argument("--n_threads", type=int, default=32)
    parser.add_argument("--sampling_strategy", "-s", default=None)
    args = parser.parse_args()
    if args.check:
        check(args, n_images=3)
    else:
        train_shallow2deep(args)
