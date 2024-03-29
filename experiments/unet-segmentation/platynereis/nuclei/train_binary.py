import torch_em
from torch_em.model import UNet3d
from torch_em.data.datasets import get_platynereis_nuclei_loader


def get_model():
    model = UNet3d(
        in_channels=1,
        out_channels=1,
        initial_features=16,
        gain=2,
        final_activation="Sigmoid",
        norm=None
    )
    return model


def get_loader(path, is_train, n_samples):
    batch_size = 1
    patch_shape = [32, 256, 256]
    if is_train:
        sample_ids = [1, 3, 6, 7, 8, 9, 10, 11, 12]
    else:
        sample_ids = [2, 4]
    loader = get_platynereis_nuclei_loader(
        path, patch_shape, sample_ids,
        binary=True,
        batch_size=batch_size,
        n_samples=n_samples,
        download=True,
        shuffle=True,
        num_workers=8*batch_size,
    )
    return loader


def train_binary(args):
    model = get_model()
    train_loader = get_loader(args.input, True, n_samples=1000)
    val_loader = get_loader(args.input, False, n_samples=100)

    name = "binary_model"
    trainer = torch_em.default_segmentation_trainer(
        name=name,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=1e-4,
        mixed_precision=True,
        log_image_interval=50,
        optimizer_kwargs={"weight_decay": 0.0005}
    )

    if args.from_checkpoint:
        trainer.fit(args.n_iterations, "latest")
    else:
        trainer.fit(args.n_iterations)


def check(args, train=True, val=True, n_images=2):
    from torch_em.util.debug import check_loader
    if train:
        print("Check train loader")
        loader = get_loader(args.input, is_train=True, n_samples=100)
        check_loader(loader, n_images)
    if val:
        print("Check val loader")
        loader = get_loader(args.input, is_train=False, n_samples=100)
        check_loader(loader, n_images)


if __name__ == "__main__":
    parser = torch_em.util.parser_helper()
    args = parser.parse_args()
    if args.check:
        check(args)
    else:
        train_binary(args)
