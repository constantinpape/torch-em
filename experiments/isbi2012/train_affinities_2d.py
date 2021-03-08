import argparse

import numpy as np
import torch.nn as nn
import torch_em
from torch_em.model import UNet2d
from torch_em.model.unet import Upsampler2d

# TODO adapt the offsets for training data at native resolution
OFFSETS = [
    [-1, 0], [0, -1],
    [-3, 0], [0, -3],
    [-9, 0], [0, -9],
    [-27, 0], [0, -27]
]


def get_loader(input_path, patch_shape, roi,
               batch_size=1,):

    raw_key = 'raw'
    label_key = 'labels/gt_segmentation'

    # we add a binary target channel for foreground background segmentation
    label_transform = torch_em.transform.label.AffinityTransform(offsets=OFFSETS,
                                                                 ignore_label=None,
                                                                 add_binary_target=False,
                                                                 add_mask=True)

    return torch_em.default_segmentation_loader(
        input_path, raw_key,
        input_path, label_key,
        batch_size=batch_size,
        patch_shape=patch_shape,
        label_transform2=label_transform,
        rois=roi,
        ndim=2,
        num_workers=8*batch_size,
        shuffle=True
    )


def get_model():
    # we have 1 channel per affinty offsets and another channel
    # for foreground background segmentation
    n_out = len(OFFSETS)

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


def train_affinties(input_path):
    model = get_model()

    # shape of input patches (blocks) used for training
    patch_shape = [1, 512, 512]

    train_loader = get_loader(
        input_path,
        patch_shape=patch_shape,
        roi=np.s_[:28, :, :]
    )
    val_loader = get_loader(
        input_path,
        patch_shape=patch_shape,
        roi=np.s_[28:, :, :]
    )

    loss = torch_em.loss.LossWrapper(
        torch_em.loss.DiceLoss(),
        transform=torch_em.loss.ApplyAndRemoveMask()
    )

    trainer = torch_em.default_segmentation_trainer(
        name='affinity-model',
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss=loss,
        metric=loss,
        learning_rate=5e-5,
        mixed_precision=True,
        log_image_interval=50
    )

    trainer.fit(int(2e4))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True)
    args = parser.parse_args()
    train_affinties(args.input)
