import torch_em
from torch_em.model import UNet3d
from torch_em.data.datasets import get_kasthuri_loader


def get_loader(args, split):
    patch_shape = (64, 256, 256)

    n_samples = 500 if split == "train" else 25
    sampler = torch_em.data.sampler.MinForegroundSampler(min_fraction=0.05, background_id=[-1, 0])
    label_transform = torch_em.transform.label.NoToBackgroundBoundaryTransform(ndim=3, add_binary_target=True)
    loader = get_kasthuri_loader(
        args.input, split=split, label_transform=label_transform,
        batch_size=args.batch_size, patch_shape=patch_shape,
        n_samples=n_samples, ndim=3, shuffle=True,
        num_workers=12, sampler=sampler
    )
    return loader


def train_direct(args):
    name = "kasthuri-mito-3d"
    model = UNet3d(in_channels=1, out_channels=2, final_activation="Sigmoid", depth=4, initial_features=32)

    train_loader = get_loader(args, "train")
    val_loader = get_loader(args, "test")
    loss = torch_em.loss.DiceLoss()
    loss = torch_em.loss.wrapper.LossWrapper(
        loss, torch_em.loss.wrapper.MaskIgnoreLabel()
    )

    trainer = torch_em.default_segmentation_trainer(
        name, model, train_loader, val_loader,
        loss=loss, learning_rate=3.0e-4, device=args.device, log_image_interval=50
    )
    trainer.fit(args.n_iterations)


if __name__ == "__main__":
    parser = torch_em.util.parser_helper()
    args = parser.parse_args()
    train_direct(args)
