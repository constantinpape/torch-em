import os
from glob import glob

import torch_em
from torch_em.model import AnisotropicUNet

# TODO adapt the offsets for training data at native resolution
OFFSETS = [
    [-1, 0, 0], [0, -1, 0], [0, 0, -1],
    [-3, 0, 0], [0, -3, 0], [0, 0, -3],
    [-9, 0, 0], [0, -9, 0], [0, 0, -9]
]


# TODO use training data at native resolution instead and clean this up
def get_datasets():
    root1 = '/g/kreshuk/pape/Work/data/platy_training_data/mitos/10nm'
    root2 = '/g/kreshuk/pape/Work/data/platy_training_data/mitos/14nm'

    # get all the paths for the raw and gt cubes
    data_folders1 = glob(os.path.join(root1, 'gt*'))
    raw_paths = [os.path.join(df, 'raw_256.h5') for df in data_folders1
                 if os.path.exists(os.path.join(df, 'raw_256.h5'))]
    gt_paths = [os.path.join(os.path.split(rp)[0], 'mito.h5') for rp in raw_paths]
    assert len(raw_paths) == len(gt_paths)

    data_folders2 = glob(os.path.join(root2, 'c*'))
    raw_paths += [os.path.join(df, 'raw_crop_center256_256_256.h5') for df in data_folders2]
    gt_paths += [os.path.join(df, 'raw_MITO.h5') for df in data_folders2]

    assert len(raw_paths) == len(gt_paths)
    assert all(os.path.exists(path) for path in raw_paths)
    assert all(os.path.exists(path) for path in gt_paths)

    return raw_paths, gt_paths


def get_loader(split, patch_shape,
               batch_size=1, n_samples=None):

    raw_paths, label_paths = get_datasets()

    split_idx = int(0.8 * len(raw_paths))
    if split == 'train':
        raw_paths = raw_paths[:split_idx]
        label_paths = label_paths[:split_idx]
    else:
        raw_paths = raw_paths[split_idx:]
        label_paths = label_paths[split_idx:]

    raw_key = 'data'
    label_key = 'data'

    # we add a binary target channel for foreground background segmentation
    ignore_label = None  # set ignore label here
    label_transform = torch_em.transform.label.AffinityTransform(offsets=OFFSETS,
                                                                 ignore_label=ignore_label,
                                                                 add_binary_target=True,
                                                                 add_mask=True)

    return torch_em.default_segmentation_loader(
        raw_paths, raw_key,
        label_paths, label_key,
        batch_size=batch_size,
        patch_shape=patch_shape,
        label_transform2=label_transform,
        n_samples=n_samples,
        num_workers=8*batch_size,
        shuffle=True
    )


def train_affinties():

    # we have 1 channel per affinty offsets and another channel
    # for foreground background segmentation
    n_out = len(OFFSETS) + 1

    # TODO adapt scale factors for data at native resolution
    scale_factors = 4 * [
        [2, 2, 2]
    ]

    model = AnisotropicUNet(
        in_channels=1,
        out_channels=n_out,
        scale_factors=scale_factors,
        initial_features=64,
        gain=2,
        final_activation='Sigmoid',
        anisotropic_kernel=False  # set anisotropic kernels if scale factors are anisotropic
    )

    # shape of input patches (blocks) used for training
    patch_shape = [128, 128, 128]

    train_loader = get_loader(
        'train',
        patch_shape=patch_shape,
        n_samples=500
    )
    val_loader = get_loader(
        'val',
        patch_shape=patch_shape,
        n_samples=50
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

    trainer.fit(int(5e4))


def check():
    from torch_em.util.debug import check_loader
    loader = get_loader(
        'train'
    )
    check_loader(loader, 4)


if __name__ == '__main__':
    # check()
    train_affinties()
