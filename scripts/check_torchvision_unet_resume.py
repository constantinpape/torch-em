import os
import argparse

import h5py

import torch

import torch_em
from torch_em.loss import DiceLoss
from torch_em.trainer import DefaultTrainer
from torch_em.transform.label import BoundaryTransform
from torch_em.model.torchvision_unet import TorchvisionUNet2d, TorchvisionUNet3d


def get_loader(args, ndim, training):
    with h5py.File(args.data, "r") as data:
        depth = data[args.raw_key].shape[0]
    split = depth // 2
    roi = (slice(0, split) if training else slice(split, depth), slice(None), slice(None))
    return torch_em.default_segmentation_loader(
        raw_paths=args.data, raw_key=args.raw_key, label_paths=args.data, label_key=args.label_key,
        batch_size=1, patch_shape=(1, 64, 64) if ndim == 2 else (8, 32, 32), ndim=ndim,
        rois=roi, n_samples=2, num_workers=0, shuffle=training,
        label_transform=BoundaryTransform(ndim=ndim),
    )


def check_resume(args, ndim):
    model_class = TorchvisionUNet2d if ndim == 2 else TorchvisionUNet3d
    model = model_class(
        backbone="resnet18" if ndim == 2 else "r3d_18", in_channels=1, out_channels=1,
        depth=2, initial_features=4, gain=3, pretrained=False, final_activation="Sigmoid",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    name = f"torchvision-unet-{ndim}d-resume"
    trainer = DefaultTrainer(
        name=name, model=model, train_loader=get_loader(args, ndim, True),
        val_loader=get_loader(args, ndim, False), loss=DiceLoss(), metric=DiceLoss(),
        optimizer=optimizer, lr_scheduler=scheduler, device="cpu", mixed_precision=False,
        compile_model=False, logger=None, save_root=args.output,
    )
    trainer.fit(2)
    checkpoint_folder = os.path.join(args.output, "checkpoints", name)
    restored = DefaultTrainer.from_checkpoint(checkpoint_folder, name="latest", device="cpu")
    assert restored.iteration == trainer.iteration == 2
    assert restored.model.init_kwargs == model.init_kwargs
    torch.testing.assert_close(restored.model.state_dict(), model.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(restored.optimizer.state_dict(), optimizer.state_dict(), rtol=0, atol=0)
    assert restored.lr_scheduler.state_dict() == scheduler.state_dict()
    assert restored.train_loader.dataset.raw.shape == trainer.train_loader.dataset.raw.shape
    restored.fit(2)
    assert restored.iteration == 4
    assert any(
        not torch.equal(before, after)
        for before, after in zip(model.parameters(), restored.model.parameters())
    )
    for state in restored.optimizer.state.values():
        assert state["step"].item() == 4
    print(f"{ndim}D: restored model, optimizer, scheduler, and loaders; resumed from iteration 2 to 4.")
    print(f"Checkpoint: {checkpoint_folder}/latest.pt")


def main():
    parser = argparse.ArgumentParser(description="Check torchvision U-Net trainer resumption on annotated HDF5 data.")
    parser.add_argument("--data", required=True, help="Path to an annotated HDF5 volume.")
    parser.add_argument("--raw-key", default="volumes/raw")
    parser.add_argument("--label-key", default="volumes/labels/neuron_ids")
    parser.add_argument("--output", required=True, help="Directory for training checkpoints.")
    args = parser.parse_args()
    for ndim in (2, 3):
        check_resume(args, ndim)


if __name__ == "__main__":
    main()
