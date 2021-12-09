import torch_em
from torch_em.data.datasets import get_mitoem_loader
from torch_em.model import AnisotropicUNet


def get_loader(input_path, samples, splits, patch_shape,
               batch_size=1, n_samples=None):
    sampler = torch_em.data.MinForegroundSampler(min_fraction=0.05, p_reject=.75)
    return get_mitoem_loader(input_path, patch_shape, splits, samples,
                             batch_size=batch_size, download=True,
                             boundaries=True, sampler=sampler)


def get_model(large_model):
    n_out = 2
    if large_model:
        print("Using large model")
        model = AnisotropicUNet(
            scale_factors=[
                [1, 2, 2],
                [1, 2, 2],
                [2, 2, 2],
                [2, 2, 2],
                [2, 2, 2]
            ],
            in_channels=1,
            out_channels=n_out,
            initial_features=128,
            gain=2,
            final_activation="Sigmoid"
        )
    else:
        print("Using vanilla model")
        model = AnisotropicUNet(
            scale_factors=[
                [1, 2, 2],
                [1, 2, 2],
                [2, 2, 2],
                [2, 2, 2]
            ],
            in_channels=1,
            out_channels=n_out,
            initial_features=64,
            gain=2,
            final_activation="Sigmoid"
        )
    return model


def train_boundaries(args, samples):
    large_model = bool(args.large_model)
    model = get_model(large_model)

    # patch shapes:
    if large_model:
        # largest possible shape for A100 with mixed training and large model
        patch_shape = [32, 256, 256]
    else:
        # largest possible shape for 2080Ti with mixed training
        # patch_shape = [32, 320, 320]
        patch_shape = [32, 256, 256]

    splits = ["train", "val"] if args.train_on_val else ["train"]
    train_loader = get_loader(args.input, samples, splits, patch_shape, batch_size=args.batch_size, n_samples=1000)
    splits = ["val"]
    val_loader = get_loader(args.input, samples, splits, patch_shape, batch_size=args.batch_size, n_samples=100)

    tag = "large" if large_model else "default"
    if args.train_on_val:
        tag += "_train_on_val"
    name = f"affinity_model_{tag}_{'_'.join(samples)}"
    trainer = torch_em.default_segmentation_trainer(
        name=name,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=1e-4,
        mixed_precision=True,
        log_image_interval=50
    )

    if args.from_checkpoint:
        trainer.fit(args.n_iterations, "latest")
    else:
        trainer.fit(args.n_iterations)


def check(input_path, samples, train=True, val=True, n_images=5):
    from torch_em.util.debug import check_loader
    patch_shape = [32, 256, 256]
    if train:
        print("Check train loader")
        loader = get_loader(input_path, samples, splits=["train"], patch_shape=patch_shape)
        check_loader(loader, n_images)
    if val:
        print("Check val loader")
        loader = get_loader(input_path, samples, splits=["val"], patch_shape=patch_shape)
        check_loader(loader, n_images)


if __name__ == "__main__":
    parser = torch_em.util.parser_helper()
    parser.add_argument("--samples", type=str, nargs="+", default=["human", "rat"])
    parser.add_argument("--large_model", "-l", type=int, default=0)
    parser.add_argument("--train_on_val", type=int, default=0)
    args = parser.parse_args()

    samples = args.samples
    samples.sort()
    if args.check:
        check(args.input, samples, train=True, val=True)
    else:
        train_boundaries(args, samples)
