import torch
import torch_em
from torch_em.model import UNet2d
from torch_em.data.datasets import get_covid_if_loader


def train_boundaries(args):
    model = UNet2d(in_channels=1, out_channels=2, initial_features=64, final_activation="Sigmoid")

    patch_shape = (512, 512)
    # use the first 5 images for validation
    train_loader = get_covid_if_loader(
        args.input, patch_shape, sample_range=(5, None),
        download=True, boundaries=True, batch_size=args.batch_size
    )
    val_loader = get_covid_if_loader(
        args.input, patch_shape, sample_range=(0, 5),
        boundaries=True, batch_size=args.batch_size
    )
    loss = torch_em.loss.DiceLoss()

    trainer = torch_em.default_segmentation_trainer(
        name="covid-if-boundary-model",
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
    train_boundaries(args)
