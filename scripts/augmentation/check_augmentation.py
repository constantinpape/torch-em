import napari
import numpy as np
import h5py
import kornia

from torch_em.transform.augmentation import KorniaAugmentationPipeline, get_augmentations
from torch_em.transform.augmentation import RandomElasticDeformation

pr = '/g/schwab/hennies/project_segmentation_paper/ds_sbem-6dpf-1-whole/seg_210122_mito/seg_10nm/gt_cubes/gt000/raw_256.h5'
pgt = '/g/schwab/hennies/project_segmentation_paper/ds_sbem-6dpf-1-whole/seg_210122_mito/seg_10nm/gt_cubes/gt000/mito.h5'

bb = np.s_[:32, :128, :128]
with h5py.File(pr, 'r') as f:
    raw = f['data'][bb].astype('float32')
with h5py.File(pgt, 'r') as f:
    seg = f['data'][bb]


def check_kornia_augmentation():
    rot = kornia.augmentation.RandomRotation(
        degrees=90., p=1.
    )

    trafo = KorniaAugmentationPipeline(
        rot
    )

    transformed_raw, transformed_seg = trafo(raw, seg)
    transformed_raw = transformed_raw.numpy().squeeze()
    transformed_seg = transformed_seg.numpy().squeeze()

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(raw)
        viewer.add_image(transformed_raw)
        viewer.add_labels(seg)
        viewer.add_labels(transformed_seg)


def check_default_augmentation():
    trafo = get_augmentations()

    transformed_raw, transformed_seg = trafo(raw, seg)
    transformed_raw = transformed_raw.numpy().squeeze()
    transformed_seg = transformed_seg.numpy().squeeze()

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(raw)
        viewer.add_image(transformed_raw)
        viewer.add_labels(seg)
        viewer.add_labels(transformed_seg)


def check_elastic_2d():
    import torch
    raw_ = raw[0]
    traw = torch.from_numpy(raw_[None, None])

    # trafo = RandomElasticDeformation(alpha=(1., 1.), p=1)
    # transformed_raw = trafo(traw)

    # noise_shape = (1, 2) + raw_.shape
    # noise = torch.zeros(noise_shape)
    amp = 1. / raw.shape[0]
    noise = np.concatenate([np.random.uniform(-amp, amp, traw.shape),
                            np.random.uniform(-amp, amp, traw.shape)], axis=1).astype('float32')
    noise = torch.from_numpy(noise)

    alpha = 1.
    transformed_raw = kornia.geometry.transform.elastic_transform2d(traw, noise, alpha=(alpha, alpha))

    transformed_raw = transformed_raw.numpy().squeeze()
    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(raw_)
        viewer.add_image(transformed_raw)


if __name__ == '__main__':
    # check_kornia_augmentation()
    # check_default_augmentation()
    check_elastic_2d()
