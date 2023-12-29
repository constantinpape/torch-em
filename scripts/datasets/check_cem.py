import os
import imageio.v3 as imageio
from glob import glob

import numpy as np
import torch_em
from torch_em.data.datasets import cem
from torch_em.util.debug import check_loader


def get_all_shapes():
    # Get the shape for the 3d datasets (id: 1-6)
    data_root = "./data/10982/data/mito_benchmarks"
    i = 1
    for root, dirs, files in os.walk(data_root):
        dirs.sort()
        for ff in files:
            if ff.endswith("em.tif"):
                shape = imageio.imread(os.path.join(root, ff)).shape
                print(i, ":", ff, ":", shape)
                i += 1

    # Get the shape for the 2d dataset (id: 7)
    data_root = "./data/10982/data/tem_benchmark/images"

    shapes_2d = []
    for image in glob(os.path.join(data_root, "*.tiff")):
        shape = imageio.imread(image).shape
        shapes_2d.append(shape)
    print(i, ":", set(shapes_2d))


def check_benchmark_loaders():
    for dataset_id in range(1, 8):
        print("Check benchmark dataset", dataset_id)
        full_shape = cem.BENCHMARK_SHAPES[dataset_id]
        if dataset_id == 7:
            patch_shape = full_shape
        else:
            patch_shape = (1,) + full_shape[1:]
        loader = cem.get_benchmark_loader(
            "./data", dataset_id=dataset_id, batch_size=1, patch_shape=patch_shape, ndim=2
        )
        check_loader(loader, 4, instance_labels=True)


def check_mitolab_loader():
    val_fraction = 0.1
    train_loader = cem.get_mitolab_loader(
        "./data", split="train", batch_size=1, shuffle=True,
        sampler=torch_em.data.sampler.MinInstanceSampler(),
        val_fraction=val_fraction,
    )
    print("Checking train loader ...")
    check_loader(train_loader, 8, instance_labels=True)
    print("... done")

    val_loader = cem.get_mitolab_loader(
        "./data", split="val", batch_size=1, shuffle=True,
        sampler=torch_em.data.sampler.MinInstanceSampler(),
        val_fraction=val_fraction,
    )
    print("Checking val loader ...")
    check_loader(val_loader, 8, instance_labels=True)
    print("... done")


def analyse_mitolab():
    data_root = "data/11037/cem_mitolab"
    folders = glob(os.path.join(data_root, "*"))

    n_datasets = len(folders)

    n_images = 0
    n_images_with_labels = 0

    for folder in folders:
        assert os.path.isdir(folder)
        images = sorted(glob(os.path.join(folder, "images", "*.tiff")))
        labels = sorted(glob(os.path.join(folder, "masks", "*.tiff")))

        n_images += len(images)
        n_labels = [len(np.unique(imageio.imread(lab))) for lab in labels]
        n_images_with_labels += sum([n_lab > 1 for n_lab in n_labels])

        # print(folder)
        # this_shapes = [imageio.imread(im).shape for im in images]
        # print(set(this_shapes))

    print(n_datasets)
    print(n_images)
    print(n_images_with_labels)


def main():
    # get_all_shapes()
    # check_benchmark_loaders()
    check_mitolab_loader()
    # analyse_mitolab()


if __name__ == "__main__":
    main()
