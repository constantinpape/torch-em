import numpy as np
import torch_em
from torch_em.model import UNet2d
from torch_em.data.datasets import get_vnc_mito_loader


def get_loader(args, split):
    patch_shape = (1, 512, 512)

    roi = np.s_[:18, :, :] if split == "train" else np.s_[18:, :, :]
    n_samples = 500 if split == "train" else 25

    loader = get_vnc_mito_loader(
        args.input, boundaries=True,
        batch_size=args.batch_size, patch_shape=patch_shape,
        n_samples=n_samples, ndim=2, shuffle=True,
        num_workers=12, rois=roi
    )
    return loader


def train_direct(args):
    name = "em-mitochondria"
    model = UNet2d(in_channels=1, out_channels=2, final_activation="Sigmoid", depth=4, initial_features=64)

    train_loader = get_loader(args, "train")
    val_loader = get_loader(args, "val")

    trainer = torch_em.default_segmentation_trainer(
        name, model, train_loader, val_loader, learning_rate=1.0e-4, device=args.device, log_image_interval=50
    )
    trainer.fit(args.n_iterations)


if __name__ == "__main__":
    parser = torch_em.util.parser_helper()
    args = parser.parse_args()
    train_direct(args)
