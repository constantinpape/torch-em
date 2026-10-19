"""Check downstream UNETR loading and inference with real images and checkpoints."""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import imageio.v3 as imageio

import torch

import torch_em
from torch_em.transform.raw import normalize_percentile


def load_model(args):
    if args.package == "micro-sam":
        from micro_sam.v1.util import get_sam_model
        from micro_sam.v1.instance_segmentation import get_unetr

        _, sam = get_sam_model(
            model_type=args.model_type, checkpoint_path=args.encoder_checkpoint, return_sam=True, device="cpu"
        )
        state = torch.load(args.decoder_checkpoint, map_location="cpu", weights_only=False)
        return get_unetr(sam.image_encoder, decoder_state=state, device="cpu")

    if args.package == "micro-sam2":
        if args.semantic:
            from micro_sam.v2.training.training import _build_semantic_sam2_model

            return _build_semantic_sam2_model(
                args.model_type, "cpu", num_classes=args.num_classes, init_decoder=args.init_decoder
            )
        from micro_sam.v2.util import get_sam2_model
        from micro_sam.v2.instance_segmentation import get_decoder

        sam = get_sam2_model(model_type=args.model_type, checkpoint_path=args.encoder_checkpoint, device="cpu")
        return get_decoder(
            model_type=args.model_type, checkpoint=args.decoder_checkpoint, encoder=sam.image_encoder, device="cpu"
        )

    sys.path.insert(0, str(Path(args.repo).resolve()))
    from micro_match.util import get_unetr_model

    model = get_unetr_model(
        ndim=args.ndim, backbone=args.backbone, model_type=args.model_type,
        out_channels=1, init_decoder=args.init_decoder
    )
    # Compare fresh parameters with the same seed after encoder construction.
    torch.manual_seed(42)
    if args.init_decoder:
        model.out_conv.reset_parameters()
    else:
        for name, module in model.named_modules():
            if not name.startswith("encoder") and hasattr(module, "reset_parameters"):
                module.reset_parameters()
    return model


def load_image(args):
    raw = imageio.imread(args.input)
    if raw.ndim != 2:
        raise ValueError(f"Expected a grayscale image, got {raw.shape}.")
    raw = raw[:args.crop_size, :args.crop_size]
    if args.package == "micro-match":
        from micro_match.util import get_raw_transform

        normalizer, _ = get_raw_transform(args.backbone)
        raw = normalizer(raw)
    else:
        raw = np.clip(normalize_percentile(raw), 0, 1)
        if args.package == "micro-sam":
            raw = raw * 255
    image = torch.as_tensor(raw, dtype=torch.float32)[None, None]
    if args.ndim == 3 or args.package == "micro-sam2":
        slices = 2 if args.ndim == 3 else 1
        image = image.unsqueeze(2).repeat(1, 1, slices, 1, 1)
    return image


def check_cached_features(args, model, features, input_shape, original_shape, prediction):
    if args.package == "micro-sam":
        from micro_sam.v1.instance_segmentation import DecoderAdapter

        cached = DecoderAdapter(model)(features[0], input_shape, original_shape)
        torch.testing.assert_close(cached, prediction, rtol=1e-4, atol=1e-5)
    elif args.package == "micro-sam2":
        from micro_sam.v2.instance_segmentation import _decode_3d_feature_batch

        cached = _decode_3d_feature_batch(model, features.transpose(1, 2), original_shape[-2:], "cpu")
        torch.testing.assert_close(torch.from_numpy(cached), prediction, rtol=1e-4, atol=1e-5)


def check_semantic_inference(args, model, image, prediction):
    from micro_sam.v2.semantic_segmentation import semantic_segmentation

    raw = image[0, 0].numpy()
    logits = prediction[0].numpy()
    if args.ndim == 2:
        raw, logits = raw[0], logits[:, 0]
    segmentation = semantic_segmentation(
        model, raw, tile_shape=raw.shape, halo=(0,) * raw.ndim, verbose=False, normalize=lambda data: data
    )
    expected = logits.argmax(axis=0) if args.num_classes > 1 else (logits[0] > 0).astype("uint8")
    np.testing.assert_array_equal(segmentation, expected)


def check_decoder_weights(args, model):
    if args.package != "micro-match" or not args.init_decoder:
        return None
    from micro_match.util import _get_checkpoint

    _, decoder_path = _get_checkpoint(args.backbone, args.model_type, return_decoder_path=True)
    checkpoint = torch.load(decoder_path, map_location="cpu", weights_only=False)
    for key in ("model_state", "model", "state_dict"):
        if key in checkpoint:
            checkpoint = checkpoint[key]
    matched = 0
    for key, value in model.state_dict().items():
        if key.startswith(("encoder.", "out_conv.")) or key not in checkpoint:
            continue
        if value.shape != checkpoint[key].shape:
            continue
        torch.testing.assert_close(
            value, checkpoint[key], msg=f"The pretrained decoder parameter '{key}' was not loaded."
        )
        matched += 1
    if matched == 0:
        raise AssertionError("No pretrained decoder parameters matched the downstream model.")
    return matched


def check_gradients(args, model, image):
    model.train()
    model.zero_grad(set_to_none=True)
    prediction = model(image)
    if args.semantic:
        from torch_em.loss.dice import BCEDiceLossWithLogits

        from micro_sam.v2.loss.semantic_loss import CustomCombinedLoss

        # Intensity bins test the loss interface without semantic annotations.
        if args.num_classes == 1:
            target = (image[:, :1] > 0.5).float()
            loss = BCEDiceLossWithLogits()(prediction, target)
        else:
            target = (image[:, :1] * args.num_classes).long().clamp(max=args.num_classes - 1)
            loss = CustomCombinedLoss(num_classes=args.num_classes)(prediction, target)
    else:
        loss = prediction.square().mean()
    assert torch.isfinite(loss)
    loss.backward()
    counts = {"encoder": 0, "decoder": 0}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        assert torch.isfinite(parameter.grad).all(), f"Nonfinite gradient for '{name}'."
        if torch.count_nonzero(parameter.grad) > 0:
            counts["encoder" if name.startswith("encoder.") else "decoder"] += 1
    assert all(counts.values()), f"Missing encoder or decoder gradients: {counts}."
    before_step = model.out_conv.weight.detach().clone()
    torch.optim.SGD(model.parameters(), lr=1e-3).step()
    assert not torch.equal(model.out_conv.weight, before_step), "The optimizer did not update the output head."
    return counts


def check_model(args):
    torch.manual_seed(42)
    model = load_model(args).eval()
    loaded_decoder_parameters = check_decoder_weights(args, model)
    if args.img_size is not None:
        if args.package == "micro-sam" or args.backbone == "sam":
            raise ValueError("Use the native image size for the SAM1 encoder.")
        model.img_size = args.img_size
        model.encoder.img_size = args.img_size
    image = load_image(args)
    original_shape = tuple(image.shape[2:])
    with torch.inference_mode():
        features, input_shape = model.encode(image)
        prediction = model.decode(features, input_shape, original_shape)
        check_cached_features(args, model, features, input_shape, original_shape, prediction)
        if args.semantic:
            assert model.final_activation is None
            check_semantic_inference(args, model, image, prediction)
    assert tuple(prediction.shape[2:]) == original_shape
    assert torch.isfinite(prediction).all()
    state_shapes = {key: tuple(value.shape) for key, value in model.state_dict().items()}
    result = {
        "prediction": prediction.cpu(), "state_shapes": state_shapes, "encoder_width": model.embed_dim,
        "decoder_width": model.out_conv.in_channels,
    }
    if args.reference:
        reference = torch.load(args.reference, map_location="cpu", weights_only=False)
        assert result["state_shapes"] == reference["state_shapes"]
        torch.testing.assert_close(result["prediction"], reference["prediction"], rtol=1e-4, atol=1e-5)
    if args.output:
        torch.save(result, args.output)
    gradient_counts = check_gradients(args, model, image) if args.check_gradients else None
    return {
        "status": "passed", "package": args.package, "ndim": args.ndim, "model_type": args.model_type,
        "encoder_width": model.embed_dim, "decoder_width": model.out_conv.in_channels,
        "encoder_img_size": model.encoder.img_size,
        "input_shape": list(image.shape), "output_shape": list(prediction.shape),
        "checked_decoder_parameters": loaded_decoder_parameters,
        "parameters_with_gradients": gradient_counts,
        "semantic": args.semantic, "output_channels": prediction.shape[1],
        "downstream_repo": str(Path(args.repo).resolve()) if args.repo else None,
        "reference_matched": bool(args.reference), "torch_em": str(torch_em.__file__),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", choices=("micro-sam", "micro-sam2", "micro-match"), required=True)
    parser.add_argument("--repo", help="The micro-match checkout to test.")
    parser.add_argument("--backbone", default="sam")
    parser.add_argument("--model-type", required=True)
    parser.add_argument("--ndim", type=int, choices=(2, 3), default=2)
    parser.add_argument("--input", required=True)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--img-size", type=int, help="The input size for encoders that support dynamic image sizes.")
    parser.add_argument("--encoder-checkpoint")
    parser.add_argument("--decoder-checkpoint")
    parser.add_argument("--init-decoder", action="store_true")
    parser.add_argument("--semantic", action="store_true", help="Check the micro-sam2 semantic training builder.")
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--check-gradients", action="store_true")
    parser.add_argument("--reference")
    parser.add_argument("--output")
    parser.add_argument("--report")
    args = parser.parse_args()
    if args.package == "micro-match" and args.repo is None:
        parser.error("--repo is required for micro-match.")
    if args.semantic and args.package != "micro-sam2":
        parser.error("--semantic is only supported for micro-sam2.")
    needs_checkpoints = args.package != "micro-match" and not args.semantic
    if needs_checkpoints and (args.encoder_checkpoint is None or args.decoder_checkpoint is None):
        parser.error("--encoder-checkpoint and --decoder-checkpoint are required for micro-sam models.")
    try:
        result = check_model(args)
    except Exception as error:
        result = {"status": "failed", "error": f"{type(error).__name__}: {error}", "torch_em": str(torch_em.__file__)}
        if args.report:
            Path(args.report).write_text(json.dumps(result, indent=2) + "\n")
        raise
    print(json.dumps(result, indent=2))
    if args.report:
        Path(args.report).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
