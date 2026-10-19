import torch

from torch_em.model.unetr import get_unetr_model

from micro_sam.v2.util import _get_checkpoint
from micro_sam.v1.util import _download_sam_model


def get_input(ndim, device, max_value):
    shape = (1, 1, 512, 512) if ndim == 2 else (1, 1, 4, 512, 512)
    return max_value * torch.rand(shape, device=device)


def load_sam1(ndim):
    """Load the SAM encoder in UNETR with pretrained weights."""
    checkpoint, _, _ = _download_sam_model("vit_b")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = get_unetr_model(
        ndim=ndim, backbone="sam", model_type="vit_b", encoder_checkpoint=checkpoint,
        out_channels=3, final_activation="Sigmoid",
    ).to(device)

    # SAM expects inputs in the range [0, 255].
    x = get_input(ndim, device, max_value=255)
    with torch.no_grad():
        y = model(x)
    print(x.shape, y.shape)
    print("UNETR with the SAM encoder loaded the weights from", checkpoint)


def load_sam2(ndim):
    """Load the SAM2 encoder in UNETR with pretrained weights."""
    checkpoint = _get_checkpoint("hvit_t")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = get_unetr_model(
        ndim=ndim, backbone="sam2", model_type="hvit_t", encoder_checkpoint=checkpoint,
        out_channels=3, final_activation="Sigmoid",
    ).to(device)

    x = get_input(ndim, device, max_value=1)
    with torch.no_grad():
        y = model(x)
    print(x.shape, y.shape)
    print("UNETR with the SAM2 encoder loaded the weights from", checkpoint)


def load_sam3(ndim):
    """Load the SAM3 encoder in UNETR without pretrained weights."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = get_unetr_model(
        ndim=ndim, backbone="sam3", model_type="vit_pe", out_channels=3, final_activation="Sigmoid"
    ).to(device)

    x = get_input(ndim, device, max_value=1)
    with torch.no_grad():
        y = model(x)
    print(x.shape, y.shape)
    print("UNETR with the SAM3 encoder was created.")


def main():
    load_sam1(ndim=2)
    load_sam1(ndim=3)

    load_sam2(ndim=2)
    load_sam2(ndim=3)

    load_sam3(ndim=2)
    load_sam3(ndim=3)


if __name__ == "__main__":
    main()
