import torch
import torch_em
import torch_em.loss.spoco_loss as spoco
from torch_em.model import UNet2d
from torch_em.data.datasets import get_dsb_loader
from torch_em.trainer.spoco_trainer import SPOCOTrainer


def train_boundaries(args):
    model = UNet2d(in_channels=1, out_channels=8, initial_features=64)

    patch_shape = (1, 256, 256)
    train_loader = get_dsb_loader(
        args.input, patch_shape, split="train", download=True, batch_size=args.batch_size, label_dtype=torch.int64,
        label_transform=torch_em.transform.label_consecutive, num_workers=4,
    )
    val_loader = get_dsb_loader(
        args.input, patch_shape, split="test", batch_size=args.batch_size, label_dtype=torch.int64,
        label_transform=torch_em.transform.label_consecutive, num_workers=4,
    )

    delta_var = 0.75
    delta_dist = 2.0
    pmaps_threshold = 0.9
    aux_loss = "dice"

    loss = spoco.SPOCOLoss(delta_var, delta_dist, aux_loss=aux_loss)
    metric = spoco.SPOCOMetric(delta_dist, pmaps_threshold=pmaps_threshold)

    trainer = torch_em.default_segmentation_trainer(
        name="dsb-spoco-model",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss=loss,
        metric=metric,
        learning_rate=1e-4,
        device=args.device,
        mixed_precision=True,
        log_image_interval=50,
        trainer_class=SPOCOTrainer,
    )
    trainer.fit(iterations=args.n_iterations)


if __name__ == '__main__':
    parser = torch_em.util.parser_helper(default_batch_size=8)
    args = parser.parse_args()
    train_boundaries(args)
