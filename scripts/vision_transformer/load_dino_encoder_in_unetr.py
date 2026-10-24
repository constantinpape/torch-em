import argparse

from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

import torch

from torch_em.model.unetr import get_unetr_model

from micro_sam.util import get_model_cache_directory
from micro_sam.models.vfm import VFM_MODELS, get_vfm_model


def get_input(ndim, device):
    shape = (1, 1, 256, 256) if ndim == 2 else (1, 1, 4, 256, 256)
    return torch.rand(shape, device=device)


def get_checkpoint_from_micro_sam_cache(vfm_model_type):
    """Download a DINO model with micro-sam and get its checkpoint from the micro-sam cache.

    The micro-sam cache stores DINOv2 as the original '.pth' file, so the function returns its path.
    It stores DINOv3 as a HuggingFace 'model.safetensors' file, so the function returns the loaded state.
    """
    get_vfm_model(vfm_model_type, device="cpu")
    spec = VFM_MODELS[vfm_model_type]
    cache_dir = get_model_cache_directory(spec["cache_folder"])

    if spec["backend"] == "torch_hub":
        return str(cache_dir / "checkpoints" / f"{spec['entrypoint']}_pretrain.pth")
    return load_file(hf_hub_download(spec["repo"], "model.safetensors", cache_dir=cache_dir, local_files_only=True))


def get_dinov2_reg4_checkpoint(model_type):
    """Download a DINOv2 model with registers, e.g. 'vit_s_reg4', to the micro-sam cache and load it.

    micro-sam does not provide these models, so the function downloads them with torch.hub.
    """
    name = f"dinov2_vit{model_type.split('_')[1]}14"
    url = f"https://dl.fbaipublicfiles.com/dinov2/{name}/{name}_reg4_pretrain.pth"
    model_dir = get_model_cache_directory("dino/v2") / "checkpoints"
    return torch.hub.load_state_dict_from_url(url, model_dir=str(model_dir), map_location="cpu")


def load_dino(ndim, backbone, model_type, checkpoint):
    """Load a DINO encoder in UNETR with pretrained weights."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = get_unetr_model(
        ndim=ndim, backbone=backbone, model_type=model_type, encoder_checkpoint=checkpoint,
        out_channels=1, final_activation="Sigmoid",
    ).to(device)

    x = get_input(ndim, device)
    with torch.no_grad():
        y = model(x)
    print(x.shape, y.shape)
    source = checkpoint if isinstance(checkpoint, str) else "a loaded state"
    print(f"UNETR with the {backbone} {model_type} encoder loaded the weights from", source)


def load_dino_from_micro_sam(ndim, vfm_model_type):
    """Load a DINO encoder in UNETR with the weights of a micro-sam model, e.g. 'vit_b_dinov3'."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_type, backbone = vfm_model_type.rsplit("_", 1)

    # micro-sam wraps the DINO model, so we pass the weights of the wrapped model.
    checkpoint = get_vfm_model(vfm_model_type, device="cpu").model.state_dict()
    model = get_unetr_model(
        ndim=ndim, backbone=backbone, model_type=model_type, encoder_checkpoint=checkpoint,
        out_channels=1, final_activation="Sigmoid",
    ).to(device)

    x = get_input(ndim, device)
    with torch.no_grad():
        y = model(x)
    print(x.shape, y.shape)
    print(f"UNETR with the {backbone} {model_type} encoder loaded the weights of '{vfm_model_type}' from micro-sam")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dinov3_checkpoint", default=None, help="The path to the original DINOv3 ViT-B file of Meta.")
    args = parser.parse_args()

    # DINOv3 from Meta. Request access at https://ai.meta.com/dinov3 to get the download links.
    if args.dinov3_checkpoint is not None:
        load_dino(ndim=2, backbone="dinov3", model_type="vit_b", checkpoint=args.dinov3_checkpoint)
        load_dino(ndim=3, backbone="dinov3", model_type="vit_b", checkpoint=args.dinov3_checkpoint)

    # DINOv3 from HuggingFace, which micro-sam downloads. Request access on the HuggingFace model page.
    dinov3_checkpoint = get_checkpoint_from_micro_sam_cache("vit_b_dinov3")
    load_dino(ndim=2, backbone="dinov3", model_type="vit_b", checkpoint=dinov3_checkpoint)
    load_dino(ndim=3, backbone="dinov3", model_type="vit_b", checkpoint=dinov3_checkpoint)

    # DINOv2 is open, so micro-sam downloads it without access.
    dinov2_checkpoint = get_checkpoint_from_micro_sam_cache("vit_s_dinov2")
    load_dino(ndim=2, backbone="dinov2", model_type="vit_s", checkpoint=dinov2_checkpoint)

    dinov2_reg4_checkpoint = get_dinov2_reg4_checkpoint("vit_s_reg4")
    load_dino(ndim=2, backbone="dinov2", model_type="vit_s_reg4", checkpoint=dinov2_reg4_checkpoint)
    load_dino(ndim=3, backbone="dinov2", model_type="vit_s_reg4", checkpoint=dinov2_reg4_checkpoint)

    # Pass the weights of the DINO model that micro-sam wraps.
    load_dino_from_micro_sam(ndim=2, vfm_model_type="vit_b_dinov2")
    load_dino_from_micro_sam(ndim=3, vfm_model_type="vit_b_dinov3")


if __name__ == "__main__":
    main()
