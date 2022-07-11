import torch
import torch_em
from torch_em.model import UNet2d
from torch_em.data.datasets import get_plantseg_loader


def get_loader(split, patch_shape, batch_size, n_samples=None, roi=None):
    sampler = torch_em.data.MinForegroundSampler(min_fraction=0.1, p_reject=1.)
    return get_plantseg_loader(
        args.input, "ovules", split, patch_shape, download=True, ndim=2,
        batch_size=batch_size, sampler=sampler, n_samples=n_samples,
        num_workers=8*batch_size, shuffle=True,
        label_dtype=torch.float32,
    )


def train_contrastive(args):
    model = UNet2d(
        in_channels=1, out_channels=args.embed_dim,
        initial_features=64, gain=2, depth=4,
        final_activation=None
    )
    patch_shape = [1, 736, 688]
    # can train with larger batch sizes for scatter
    batch_size = 4 if args.impl == "scatter" else 1

    train_loader = get_loader(
        split="train",
        patch_shape=patch_shape,
        batch_size=batch_size,
        n_samples=2500
    )
    val_loader = get_loader(
        split="val",
        patch_shape=patch_shape,
        batch_size=1,
        n_samples=100
    )

    loss = torch_em.loss.ContrastiveLoss(
        delta_var=.75,
        delta_dist=2.,
        impl=args.impl
    )

    name = "embedding_model2d_" + args.impl + "_d" + str(args.embed_dim)
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

    if args.from_checkpoint:
        trainer.fit(args.iterations, "latest")
    else:
        trainer.fit(args.iterations)


def check(train=True, val=True, n_images=5):
    from torch_em.util.debug import check_loader
    patch_shape = [1, 512, 512]
    if train:
        print("Check train loader")
        loader = get_loader("train", patch_shape, batch_size=1)
        check_loader(loader, n_images)
    if val:
        print("Check val loader")
        loader = get_loader("val", patch_shape, batch_size=1)
        check_loader(loader, n_images)


if __name__ == "__main__":
    parser = torch_em.util.parser_helper()
    parser.add_argument("--impl", default="scatter")
    parser.add_argument("--iterations", "-n", type=int, default=int(1e5))
    parser.add_argument("-d", "--embed_dim", type=int, default=12)
    args = parser.parse_args()
    if args.check:
        check(train=True, val=True)
    else:
        train_contrastive(args)
