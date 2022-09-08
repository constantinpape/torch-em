# train spoco from sparse instance labels
# before you can run this you need to run 'train_embeddings.py' at least once to get the dsb data
# and then run 'create_sparse_labels.py' with the correct data path to create the sparse training data

import os

import torch
import torch_em
import torch_em.loss.spoco_loss as spoco
from torch_em.model import UNet2d
from torch_em.metric import EmbeddingMWSIOUMetric
from torch_em.trainer.spoco_trainer import SPOCOTrainer


OFFSETS = [
    [-1, 0], [0, -1],
    [-3, 0], [0, -3],
    [-9, 0], [0, -9],
    [-27, 0], [0, -27],
]


def get_loader(input_, patch_shape, split, label_fraction, batch_size, num_workers):
    image_folder = os.path.join(input_, split, "images")
    label_folder = os.path.join(input_, split, "sparse_labels", f"fraction-{label_fraction}")
    assert os.path.exists(label_folder), label_folder
    return torch_em.default_segmentation_loader(
        image_folder, "*.tif", label_folder, "*.tif",
        patch_shape=patch_shape, batch_size=batch_size,
        label_dtype=torch.int64, num_workers=num_workers,
        label_transform=torch_em.transform.label_consecutive,
    )


def main():
    parser = torch_em.util.parser_helper(default_batch_size=1)
    parser.add_argument(
        "-l", "--label_fraction", required=True, type=float
    )
    parser.add_argument(
        "-a", "--affinity_side_loss", type=int, default=0, help="Whether to also use an affnity side loss."
    )
    args = parser.parse_args()

    delta_var = 0.75
    delta_dist = 2.0
    aux_loss = "dice_aff" if bool(args.affinity_side_loss) else "dice"
    loss = spoco.SPOCOLoss(delta_var, delta_dist, aux_loss=aux_loss)

    embed_dim = 8
    metric = EmbeddingMWSIOUMetric(delta_dist, OFFSETS, min_seg_size=75)
    model = UNet2d(in_channels=1, out_channels=embed_dim, initial_features=64, depth=4)

    patch_shape = (256, 256)
    num_workers = 7
    train_loader = get_loader(
        args.input, patch_shape, "train", batch_size=args.batch_size, num_workers=num_workers,
        label_fraction=args.label_fraction
    )
    val_loader = get_loader(
        args.input, patch_shape, "test", batch_size=args.batch_size, num_workers=num_workers,
        label_fraction=args.label_fraction
    )

    name = f"sparse_embeddings_fraction-{args.label_fraction}"
    if bool(args.affinity_side_loss):
        name += "_with_affinity_side_loss"
    trainer = torch_em.default_segmentation_trainer(
        name=name,
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


if __name__ == "__main__":
    main()
