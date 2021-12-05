import torch
import torch_em
from torch_em.model import UNet2d
from torch_em.data.datasets import get_monuseg_loader


def train_affinities(args):
    model = UNet2d(in_channels=3, out_channels=2, initial_features=64, final_activation="Sigmoid")

    patch_shape = (1, 512, 512)
    train_loader = get_monuseg_loader(
        args.input, patch_shape, roi=slice(0, 27),
        download=True, boundaries=True, batch_size=args.batch_size
    )
    val_loader = get_monuseg_loader(
        args.input, patch_shape, roi=slice(27, None),
        boundaries=True, batch_size=args.batch_size
    )
    loss = torch_em.loss.DiceLoss()

    # the trainer object that handles the training details
    # the model checkpoints will be saved in "checkpoints/dsb-boundary-model"
    # the tensorboard logs will be saved in "logs/dsb-boundary-model"
    trainer = torch_em.default_segmentation_trainer(
        name="monuseg-boundaty-model",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss=loss,
        metric=loss,
        learning_rate=1e-4,
        device=torch.device("cuda"),
        mixed_precision=True,
        log_image_interval=50
    )
    trainer.fit(iterations=args.n_iterations)


if __name__ == '__main__':
    parser = torch_em.util.parser_helper(
        default_batch_size=8
    )
    args = parser.parse_args()
    train_affinities(args)
