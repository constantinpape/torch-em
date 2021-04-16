import argparse

import numpy as np
import torch.nn as nn
import torch_em
from torch_em.model import UNet2d
from torch_em.model.unet import Upsampler2d

OFFSETS = [
    [-1, 0], [0, -1],
    [-3, 0], [0, -3],
    [-9, 0], [0, -9],
    [-27, 0], [0, -27]
]

DIAG_OFFSETS = [[-2, -9],
                [-5, -7],
                [-7, -5],
                [-9, -2],
                [-9, 2],
                [-7, 5],
                [-5, 7],
                [-2, 9]]


def get_offsets(use_diagonal_offsets):
    if use_diagonal_offsets:
        offsets = OFFSETS[:4] + DIAG_OFFSETS + OFFSETS[6:]
    else:
        offsets = OFFSETS
    return offsets


def get_loader(input_path, patch_shape, roi,
               batch_size=1, use_diagonal_offsets=False):

    raw_key = 'raw'
    label_key = 'labels/gt_segmentation'

    offsets = get_offsets(use_diagonal_offsets)
    # we add a binary target channel for foreground background segmentation
    label_transform = torch_em.transform.label.AffinityTransform(offsets=offsets,
                                                                 ignore_label=None,
                                                                 add_binary_target=False,
                                                                 add_mask=True)

    return torch_em.default_segmentation_loader(
        input_path, raw_key,
        input_path, label_key,
        batch_size=batch_size,
        patch_shape=patch_shape,
        label_transform2=label_transform,
        raw_transform=lambda x: x.astype('float32') / 255.,
        rois=roi,
        ndim=2,
        num_workers=8*batch_size,
        shuffle=True
    )


def get_model(use_diagonal_offsets):
    n_out = len(get_offsets(use_diagonal_offsets))

    def my_sampler(scale_factor, inc, outc):
        return Upsampler2d(scale_factor, inc, outc, mode='bilinear')

    model = UNet2d(
        in_channels=1,
        out_channels=n_out,
        final_activation='Sigmoid',
        pooler_impl=nn.AvgPool2d,
        sampler_impl=my_sampler
    )
    return model


def train_affinties(input_path, use_diagonal_offsets):
    model = get_model(use_diagonal_offsets)

    # shape of input patches (blocks) used for training
    patch_shape = [1, 512, 512]

    train_loader = get_loader(
        input_path,
        patch_shape=patch_shape,
        roi=np.s_[:28, :, :],
        use_diagonal_offsets=use_diagonal_offsets
    )
    val_loader = get_loader(
        input_path,
        patch_shape=patch_shape,
        roi=np.s_[28:, :, :],
        use_diagonal_offsets=use_diagonal_offsets
    )

    loss = torch_em.loss.LossWrapper(
        torch_em.loss.DiceLoss(),
        transform=torch_em.loss.ApplyAndRemoveMask()
    )

    name = 'affinity-model'
    if use_diagonal_offsets:
        name += '_diagonal_offsets'
    trainer = torch_em.default_segmentation_trainer(
        name=name,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss=loss,
        metric=loss,
        learning_rate=5e-5,
        mixed_precision=True,
        log_image_interval=50
    )

    trainer.fit(int(1e4))


def print_the_offsets(use_diagonal_offsets):
    offs = get_offsets(use_diagonal_offsets)
    print(offs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-d', '--use_diagonal_offsets', type=int, default=0)
    parser.add_argument('-p', '--print_offsets', default=0)
    args = parser.parse_args()
    if bool(args.print_offsets):
        print_the_offsets(bool(args.use_diagonal_offsets))
    else:
        train_affinties(args.input, bool(args.use_diagonal_offsets))
