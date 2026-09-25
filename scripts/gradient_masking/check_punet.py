import argparse
from unittest.mock import patch
from types import SimpleNamespace
from tempfile import TemporaryDirectory

import imageio.v3 as imageio

import torch
import torch.nn.functional as F

from torch_em.loss import DiceLoss
from torch_em.model import ProbabilisticUNet
from torch_em.self_training.logger import ProbabilisticUNetTrainerLogger
from torch_em.self_training.loss import ProbabilisticUNetLoss, ProbabilisticUNetLossAndMetric


def load_data(image_path, label_path):
    image = torch.as_tensor(imageio.imread(image_path).astype("float32")[:128, :128])[None, None]
    labels = torch.as_tensor(imageio.imread(label_path).astype("int64")[:128, :128])[None, None]
    assert image.shape == labels.shape == (1, 1, 128, 128)
    image = (image - image.mean()) / image.std().clamp(min=1e-6)
    return image, labels


def check_reconstruction(image, instances, channels, use_dice):
    labels = (instances > 0).long()
    if channels > 1:
        labels = torch.where(instances > 0, instances.remainder(channels - 1) + 1, 0)
    model = ProbabilisticUNet(
        output_channels=channels, num_filters=[8, 16], latent_dim=2,
        consensus_masking=True, rl_swap=use_dice, beta=0, device="cpu"
    )
    with torch.no_grad():
        model(image, labels)
        logits = model.reconstruct(use_posterior_mean=True)

    for mask_kind in ("empty", "partial", "full"):
        mask = torch.zeros_like(labels, dtype=torch.float32)
        if mask_kind == "partial":
            mask[:, :, 16:48, 16:48] = 1
        elif mask_kind == "full":
            mask.fill_(1)
        prediction = logits.detach().requires_grad_()
        with patch.object(model, "reconstruct", return_value=prediction):
            loss = -model.elbo(labels, mask)
        loss.backward()
        outside = ~mask.bool().expand_as(prediction)
        assert torch.count_nonzero(prediction.grad[outside]) == 0
        assert torch.isfinite(loss)
        if mask_kind == "empty":
            assert loss.item() == 0
            continue
        assert torch.count_nonzero(prediction.grad[~outside]) > 0

        if use_dice:
            probabilities = prediction.sigmoid() if channels == 1 else prediction.softmax(dim=1)
            target = labels.float() if channels == 1 else F.one_hot(labels[:, 0], channels).movedim(-1, 1).float()
            if mask_kind == "partial":
                probabilities = probabilities[:, :, 16:48, 16:48]
                target = target[:, :, 16:48, 16:48]
            expected = DiceLoss()(probabilities, target)
        elif channels == 1:
            expected = (F.binary_cross_entropy_with_logits(prediction, labels.float(), reduction="none") * mask).sum()
        else:
            expected = (F.cross_entropy(prediction, labels[:, 0], reduction="none") * mask[:, 0]).sum()
        torch.testing.assert_close(loss, expected)

    with torch.no_grad(), patch.object(model, "sample", return_value=logits):
        loss, metric = ProbabilisticUNetLossAndMetric(prior_samples=2)(model, image, labels)
        probabilities = logits.sigmoid() if channels == 1 else logits.softmax(dim=1)
        target = labels.float() if channels == 1 else F.one_hot(labels[:, 0], channels).movedim(-1, 1).float()
        torch.testing.assert_close(metric, DiceLoss()(probabilities, target))
        assert torch.isfinite(loss)
    print(f"Passed reconstruction, mask gradients and validation: channels={channels}, dice={use_dice}")


def check_livecell_helpers(image):
    from experiments.probabilistic_domain_adaptation.livecell.common import get_punet, get_punet_predictions

    model = get_punet().cpu()
    prediction = get_punet_predictions(model, image)
    assert prediction.shape == image.shape
    assert torch.isfinite(prediction).all()
    print("Passed LIVECell model and prediction helpers")


def check_multiple_raters(image, instances, channels, use_dice):
    image = torch.cat([image, image.flip(-1)])
    instances = torch.cat([instances, instances.flip(-1)])
    labels = (instances > 0).long()
    if channels > 1:
        labels = torch.where(instances > 0, instances.remainder(channels - 1) + 1, 0)
    # Shift the real annotation to exercise disagreement between three test raters.
    labels = torch.cat([labels, labels.roll(3, -1), labels.roll(-3, -2)], dim=1)
    model = ProbabilisticUNet(
        output_channels=channels, num_raters=3, num_filters=[8, 16], latent_dim=2,
        consensus_masking=True, rl_swap=use_dice, beta=0, device="cpu"
    )
    with torch.no_grad():
        model(image, labels)
        logits = model.reconstruct(use_posterior_mean=True)
        assert logits.shape == (2, channels, *image.shape[2:])
        assert model.posterior.encoder.layers[0].in_channels == 1 + 3 * channels
        assert model.posterior_latent_space.mean.shape == (2, 2)

    for mask_kind in ("shared", "per_rater", "empty"):
        mask = torch.zeros_like(labels, dtype=torch.float32)
        if mask_kind != "empty":
            mask[:, :, 16:48, 16:48] = 1
        if mask_kind == "per_rater":
            mask[:, -1] = 0
        elif mask_kind == "shared":
            mask = mask[:, :1]
        prediction = logits.detach().requires_grad_()
        with patch.object(model, "reconstruct", return_value=prediction):
            loss = -model.elbo(labels, mask)
        loss.backward()
        outside = ~mask.bool().any(dim=1, keepdim=True).expand_as(prediction)
        assert torch.count_nonzero(prediction.grad[outside]) == 0
        if mask_kind == "empty":
            assert loss.item() == 0
            continue
        assert torch.count_nonzero(prediction.grad[~outside]) > 0

        crop = logits[:, :, 16:48, 16:48]
        expected = []
        for rater in range(3):
            target = labels[:, rater:rater + 1, 16:48, 16:48]
            if use_dice:
                probabilities = crop.sigmoid() if channels == 1 else crop.softmax(dim=1)
                if channels > 1:
                    target = F.one_hot(target[:, 0], channels).movedim(-1, 1)
                value = DiceLoss()(probabilities, target.float())
            elif channels == 1:
                value = F.binary_cross_entropy_with_logits(crop, target.float(), reduction="sum") / len(image)
            else:
                value = F.cross_entropy(crop, target[:, 0], reduction="sum") / len(image)
            expected.append(value * (mask_kind != "per_rater" or rater != 2))
        torch.testing.assert_close(loss, torch.stack(expected).mean())

    model.beta = 1
    model.zero_grad()
    training_loss = ProbabilisticUNetLoss()(model, image, labels)
    training_loss.backward()
    for module in (model.posterior, model.prior, model.unet, model.fcomb):
        gradients = [parameter.grad for parameter in module.parameters() if parameter.requires_grad]
        assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
        assert any(torch.count_nonzero(gradient) > 0 for gradient in gradients)

    with torch.no_grad(), patch.object(model, "sample", return_value=logits):
        loss, metric = ProbabilisticUNetLossAndMetric(prior_samples=2)(model, image, labels)
        probabilities = logits.sigmoid() if channels == 1 else logits.softmax(dim=1)
        metrics = []
        for target in labels.split(1, dim=1):
            if channels > 1:
                target = F.one_hot(target[:, 0], channels).movedim(-1, 1)
            metrics.append(DiceLoss()(probabilities, target.float()))
        torch.testing.assert_close(metric, torch.stack(metrics).mean())
        assert torch.isfinite(loss)
        assert torch.isfinite(model.elbo(labels, analytic_kl=False))
    with TemporaryDirectory() as log_dir:
        trainer = SimpleNamespace(name="punet-raters", log_image_interval=1)
        logger = ProbabilisticUNetTrainerLogger(trainer, save_root=log_dir)
        try:
            logger.add_image(image, labels, [logits.detach()], "validation", 0)
        finally:
            logger.tb.close()
    print(f"Passed joint posterior, rater losses, masks, training and validation: channels={channels}, dice={use_dice}")


def main():
    parser = argparse.ArgumentParser(description="Check probabilistic U-Net losses with real images and labels.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--labels", required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(2)
    image, instances = load_data(args.image, args.labels)
    for channels in (1, 3):
        for use_dice in (False, True):
            check_reconstruction(image[:, :, :64, :64], instances[:, :, :64, :64], channels, use_dice)
            check_multiple_raters(image[:, :, :64, :64], instances[:, :, :64, :64], channels, use_dice)
    check_livecell_helpers(image)


if __name__ == "__main__":
    main()
