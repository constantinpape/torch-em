import torch_em
from torch_em.model import UNet2d
from torch_em.data.datasets import get_axondeepseg_loader


def normalize_datasets(datasets):
    datasets = list(sorted(args.datasets))
    return datasets


def get_loader(args, patch_shape, split):
    data_fraction = 0.85 if split == "train" else 0.15
    datasets = normalize_datasets(args.datasets)
    return get_axondeepseg_loader(args.input, datasets, download=True, one_hot_encoding=[0, 1, 2],
                                  data_fraction=data_fraction,
                                  patch_shape=patch_shape,
                                  shuffle=True, ndim=2,
                                  batch_size=args.batch_size,
                                  split=split)


def train_semantic_network(args):
    # we have three output channels: bg, myelin & axon (background is learned implicitly)
    n_out = 3
    # could also try a softmax here
    model = UNet2d(in_channels=1, out_channels=n_out, final_activation="Sigmoid")

    # shape of input patches used for training
    patch_shape = [576, 576]

    train_loader = get_loader(args, patch_shape, "train")
    val_loader = get_loader(args, patch_shape, "val")

    datasets = normalize_datasets(args.datasets)
    name = f"dice-axon-model-{'_'.join(datasets)}"
    trainer = torch_em.default_segmentation_trainer(
        name=name,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=1e-4,
        mixed_precision=True,
        log_image_interval=50,
        device=args.device,
    )
    trainer.fit(args.n_iterations)


def check(args, n_images=2):
    from torch_em.util.debug import check_loader
    patch_shape = [768, 768]

    print("Check train loader")
    loader = get_loader(args, patch_shape, "train")
    check_loader(loader, n_images)

    print("Check val loader")
    loader = get_loader(args, patch_shape, "val")
    check_loader(loader, n_images)


if __name__ == "__main__":
    parser = torch_em.util.parser_helper(default_batch_size=8)
    parser.add_argument("-d", "--datasets", nargs="+", default=["sem", "tem"])
    args = parser.parse_args()
    if args.check:
        check(args)
    else:
        train_semantic_network(args)
