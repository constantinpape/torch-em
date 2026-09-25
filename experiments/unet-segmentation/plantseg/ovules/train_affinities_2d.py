import torch
import torch_em
from torch_em.model import UNet2d
from torch_em.data.datasets import get_plantseg_loader

OFFSETS = [
    [-1, 0], [0, -1],
    [-3, 0], [0, -3],
    [-9, 0], [0, -9],
    [-27, 0], [0, -27]
]


def get_loader(split, patch_shape, batch_size, n_samples=None, roi=None):
    sampler = torch_em.data.MinForegroundSampler(min_fraction=0.1, p_reject=1.)
    return get_plantseg_loader(
        args.input, "ovules", split, patch_shape, download=True, offsets=OFFSETS, ndim=2,
        batch_size=batch_size, sampler=sampler, n_samples=n_samples,
        num_workers=8*batch_size, shuffle=True,
        label_dtype=torch.float32,
    )


def train_affinties(args):
    model = UNet2d(
        in_channels=1, out_channels=len(OFFSETS),
        initial_features=64, gain=2, depth=4,
        final_activation="Sigmoid"
    )
    patch_shape = [1, 512, 512]

    train_loader = get_loader(
        split="train", patch_shape=patch_shape, batch_size=args.batch_size, n_samples=2500
    )
    val_loader = get_loader(
        split="val", patch_shape=patch_shape, batch_size=args.batch_size, n_samples=100
    )

    loss = torch_em.loss.LossWrapper(
        torch_em.loss.DiceLoss(),
        transform=torch_em.loss.ApplyAndRemoveMask()
    )

    name = "affinity_model2d"
    trainer = torch_em.default_segmentation_trainer(
        name=name,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss=loss,
        metric=loss,
        learning_rate=1e-4,
        mixed_precision=True,
        log_image_interval=50
    )

    if args.from_checkpoint:
        trainer.fit(args.n_iterations, "latest")
    else:
        trainer.fit(args.n_iterations)


if __name__ == "__main__":
    parser = torch_em.util.parser_helper()
    args = parser.parse_args()
    train_affinties(args)
