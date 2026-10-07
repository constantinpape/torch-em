import os
import argparse

import numpy as np

import torch

import torch_em
from torch_em.util import load_model
from torch_em.data import MinForegroundSampler
from torch_em.model.cryozoo import CryoSiamEncoder, CryoZooUNet, get_cryozoounet, load_cryosiam_encoder_state


def cryosiam_raw_transform(raw, lower=1.0, upper=99.0):
    """Invert the contrast and scale the intensities to [0, 1], as the CryoSiam pretraining expects."""
    raw = -raw.astype("float32")
    lo, hi = np.percentile(raw, lower), np.percentile(raw, upper)
    return np.clip((raw - lo) / (hi - lo + 1e-7), 0, 1)


def get_loaders(args):
    kwargs = {
        "raw_paths": args.raw_path, "raw_key": "data", "label_paths": args.label_path, "label_key": "data",
        "patch_shape": args.patch_shape, "batch_size": args.batch_size, "ndim": 3, "is_seg_dataset": True,
        "raw_transform": cryosiam_raw_transform, "sampler": MinForegroundSampler(min_fraction=0.01),
        "num_workers": 4,
    }
    z = args.val_start
    train_loader = torch_em.default_segmentation_loader(
        rois=np.s_[:z], n_samples=args.batch_size * args.iterations, shuffle=True, **kwargs
    )
    val_loader = torch_em.default_segmentation_loader(rois=np.s_[z:], n_samples=4 * args.batch_size, **kwargs)
    return train_loader, val_loader


def check_pretrained_init(model, checkpoint):
    expected = load_cryosiam_encoder_state(checkpoint)
    actual = model.encoder.resnet.state_dict()
    assert expected.keys() == actual.keys(), "The encoder keys do not match the checkpoint."
    assert all(torch.equal(expected[k], actual[k].cpu()) for k in expected), "The encoder weights do not match."
    print(f"All {len(expected)} encoder tensors match the CryoSiam checkpoint.")


def check_reconstruction(model, checkpoint_dir, x, device):
    rebuilt = CryoZooUNet(**model.init_kwargs)
    print("Rebuilt the model from its init_kwargs:", type(rebuilt).__name__)

    reloaded = load_model(checkpoint_dir, name="latest", device=device).eval()
    model.eval()
    with torch.no_grad():
        y, y_reloaded = model(x), reloaded(x)
    assert torch.allclose(y, y_reloaded, atol=1e-5), "The reloaded model gives different predictions."
    print(f"The model reloaded from {checkpoint_dir} gives the same predictions, output shape {tuple(y.shape)}.")


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = CryoSiamEncoder.get_checkpoint(args.checkpoint_path)
    model = get_cryozoounet(out_channels=1, checkpoint_path=checkpoint, final_activation="Sigmoid")
    check_pretrained_init(model, checkpoint)

    train_loader, val_loader = get_loaders(args)
    trainer = torch_em.default_segmentation_trainer(
        name=args.name, model=model, train_loader=train_loader, val_loader=val_loader,
        learning_rate=1e-4, device=device, save_root=args.save_root, mixed_precision=device == "cuda",
        log_image_interval=50,
    )
    trainer.fit(iterations=args.iterations)

    x, _ = next(iter(val_loader))
    checkpoint_dir = os.path.join(args.save_root, "checkpoints", args.name)
    check_reconstruction(trainer.model, checkpoint_dir, x.to(device), device)


def main():
    parser = argparse.ArgumentParser(description="Finetune a CryoZooUNet with the pretrained CryoSiam encoder.")
    parser.add_argument("--raw_path", required=True, help="The tomogram in mrc format.")
    parser.add_argument("--label_path", required=True, help="The binary labels in mrc format.")
    parser.add_argument("--save_root", required=True, help="The folder for the checkpoints and logs.")
    parser.add_argument("--name", default="cryozoounet", help="The name of the checkpoint.")
    parser.add_argument("--checkpoint_path", help="The CryoSiam checkpoint. By default, the script downloads it.")
    parser.add_argument("--patch_shape", type=int, nargs=3, default=[32, 128, 128], help="It must be divisible by 8.")
    parser.add_argument("--batch_size", type=int, default=1, help="The batch size.")
    parser.add_argument("--iterations", type=int, default=1000, help="The number of training iterations.")
    parser.add_argument("--val_start", type=int, default=400, help="The first z-slice of the validation data.")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
