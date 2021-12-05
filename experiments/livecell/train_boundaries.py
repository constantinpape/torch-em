import torch
import torch_em
from torch_em.model import UNet2d
from torch_em.data.datasets import get_livecell_loader


def train_boundaries(args):
    n_out = 2
    model = UNet2d(in_channels=1, out_channels=n_out, initial_features=64,
                   final_activation="Sigmoid")

    patch_shape = (512, 512)
    train_loader = get_livecell_loader(
        args.input, patch_shape, "train",
        download=True, boundaries=True, batch_size=args.batch_size
    )
    val_loader = get_livecell_loader(
        args.input, patch_shape, "val",
        boundaries=True, batch_size=args.batch_size
    )
    loss = torch_em.loss.DiceLoss()

    trainer = torch_em.default_segmentation_trainer(
        name="livecell-boundary-model",
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
