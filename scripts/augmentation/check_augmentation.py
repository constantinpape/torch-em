import os

import h5py
import kornia
import napari
import numpy as np
import torch

import torch_em.transform.augmentation as augmentation
from torch_em.data.datasets.uro_cell import _require_urocell_data


def get_data():
    _require_urocell_data("./data", download=True)
    path = "./data/fib1-3-3-0.h5"
    assert os.path.exists(path)
    bb = np.s_[32:64, 128:256, 128:256]
    with h5py.File(path, "r") as f:
        raw = f["raw"][bb].astype("float32")
        seg = f["labels/mito"][bb]
    return raw, seg


def check_kornia_augmentation():
    raw, seg = get_data()

    rot = kornia.augmentation.RandomRotation(degrees=90.0, p=1.0)
    trafo = augmentation.KorniaAugmentationPipeline(rot)

    transformed_raw, transformed_seg = trafo(raw, seg)
    transformed_raw = transformed_raw.numpy().squeeze()
    transformed_seg = transformed_seg.numpy().squeeze()

    viewer = napari.Viewer()
    viewer.add_image(raw)
    viewer.add_image(transformed_raw)
    viewer.add_labels(seg)
    viewer.add_labels(transformed_seg)
    napari.run()


def check_default_augmentation():
    raw, seg = get_data()

    trafo = augmentation.get_augmentations()

    transformed_raw, transformed_seg = trafo(raw, seg)
    transformed_raw = transformed_raw.numpy().squeeze()
    transformed_seg = transformed_seg.numpy().squeeze().astype("uint32")

    viewer = napari.Viewer()
    viewer.add_image(raw)
    viewer.add_image(transformed_raw)
    viewer.add_labels(seg)
    viewer.add_labels(transformed_seg)
    napari.run()


def check_elastic_2d():
    raw, seg = get_data()
    raw = raw[0]
    traw = torch.from_numpy(raw[None, None])

    trafo = augmentation.RandomElasticDeformation(alpha=(1., 1.), p=1)
    transformed_raw = trafo(traw)

    # noise_shape = (1, 2) + raw_.shape
    # noise = torch.zeros(noise_shape)
    # amp = 1. / raw.shape[0]
    # noise = np.concatenate([np.random.uniform(-amp, amp, traw.shape),
    #                         np.random.uniform(-amp, amp, traw.shape)], axis=1).astype('float32')
    # noise = torch.from_numpy(noise)

    # alpha = 1.
    # transformed_raw = kornia.geometry.transform.elastic_transform2d(traw, noise, alpha=(alpha, alpha))

    transformed_raw = transformed_raw.numpy().squeeze()
    viewer = napari.Viewer()
    viewer.add_image(raw)
    viewer.add_image(transformed_raw)
    napari.run()


def check_elastic_3d():
    raw, seg = get_data()
    traw = torch.from_numpy(raw[None, None])

    trafo = augmentation.RandomElasticDeformation3D(alpha=(1., 1.), p=1)
    transformed_raw = trafo(traw)

    transformed_raw = transformed_raw.numpy().squeeze()
    viewer = napari.Viewer()
    viewer.add_image(raw)
    viewer.add_image(transformed_raw)
    napari.run()


if __name__ == "__main__":
    # check_kornia_augmentation()
    # check_default_augmentation()
    check_elastic_2d()
    # check_elastic_3d()
