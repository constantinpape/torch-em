import torch
import torch_em
from torch_em.model import UNet2d
from torch_em.data.datasets import get_dsb_loader


def train_affinties(args):
    model = UNet2d(in_channels=1, out_channels=2, initial_features=64)

    patch_shape = (1, 256, 256)
    train_loader = get_dsb_loader(
        args.input, patch_shape, split="train",
        download=True, boundaries=True, batch_size=args.batch_size
    )
    val_loader = get_dsb_loader(
        args.input, patch_shape, split="test",
        boundaries=True, batch_size=args.batch_size
    )

    loss = torch_em.loss.LossWrapper(
        torch_em.loss.DiceLoss(),
        transform=torch_em.loss.ApplyAndRemoveMask()
    )

    # the trainer object that handles the training details
    # the model checkpoints will be saved in "checkpoints/dsb-boundary-model"
    # the tensorboard logs will be saved in "logs/dsb-boundary-model"
    trainer = torch_em.default_segmentation_trainer(
        name="dsb-boundary-model",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss=loss,
        metric=loss,
        learning_rate=1e-4,
        device=torch.device('cuda'),
        mixed_precision=True,
        log_image_interval=50
    )
    trainer.fit(iterations=args.n_iterations)


if __name__ == '__main__':
    parser = torch_em.util.parser_helper(
        default_batch_size=8
    )
    args = parser.parse_args()
    train_affinties(args)
