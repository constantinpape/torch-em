import os

from elf.io import open_file
from torch_em.data.datasets import get_bioimageio_dataset_id
from torch_em.util import (add_weight_formats, export_parser_helper,
                           export_bioimageio_model, get_default_citations,
                           get_training_summary)


def _load_data(input_, organelle):
    if organelle == "cells":
        key = "volumes/raw/s1"
    else:
        key = "volumes/raw"
    with open_file(input_, "r") as f:
        ds = f[key]
        shape = ds.shape
        halo = [16, 128, 128]
        bb = tuple(slice(sh // 2 - ha, sh // 2 + ha) for sh, ha in zip(shape, halo))
        raw = ds[bb]
    return raw


def _get_name_and_description(is_aff, organelle):
    name = f"PlatynereisEM{organelle}Segmentation"
    if is_aff:
        name += "AffinityModel"
    else:
        name += "BoundaryModel"
    description = "{organelle} segmentation in EM of platynereis."
    return name, description


def _get_doc(ckpt, name, organelle):
    training_summary = get_training_summary(ckpt, to_md=True, lr=1.0e-4)
    model_tag = name.lower()
    doc = f"""# U-Net for {organelle} segmentation in platynereis

This model segments {organelle} in electron microscopy images of Platynereis dumerilii.

## Training

The network was trained on data from [Whole-body integration of gene expression and single-cell morphology](https://doi.org/10.1016/j.cell.2021.07.017).
The training script can be found [here](https://github.com/constantinpape/torch-em/tree/main/experiments/platynereis).
This folder also includes example usages of this model.

### Training Data

- Imaging modality: electron microscopy
- Dimensionality: 3D
- Source: https://doi.org/10.1038/s41592-021-01249-6

### Recommended Validation

For validation, please refer to https://github.com/constantinpape/torch-em/tree/main/experiments/platynereis.
This model can also be used in ilastik, deepimageJ or other software that supports the bioimage.io model format.

### Training Schedule

{training_summary}

## Contact

For questions or issues with this models, please reach out by:
- opening a topic with tags bioimageio and {model_tag} on [image.sc](https://forum.image.sc/)
- or creating an issue in https://github.com/constantinpape/torch-em"""
    return doc


# TODO write offsets and other mws params into the config if this is a affinity model
def export_to_bioimageio(checkpoint, output, input_, affs_to_bd, additional_formats):

    root, ckpt_name = os.path.split(checkpoint)
    organelle = os.path.split(root)[0]
    assert organelle in ("cells", "mitochondria", "nuclei"), organelle

    if input_ is None:
        input_data = None
    else:
        input_data = _load_data(input_, organelle)

    is_aff_model = "affinity" in ckpt_name
    if is_aff_model and affs_to_bd:
        if organelle in ("cells",):
            postprocessing = "affinities_to_boundaries_anisotropic"
        elif organelle in ("mitochondria", "nuclei"):
            postprocessing = "affinities_with_foreground_to_boundaries3d"
    else:
        postprocessing = None

    if is_aff_model and affs_to_bd:
        is_aff_model = False
    name, description = _get_name_and_description(is_aff_model, organelle)
    tags = ["unet", organelle, "instance-segmentation", "electron-microscopy", "platynereis"]
    tags += ["boundary-prediction"] if is_aff_model else ["affinity-prediction"]

    cite = get_default_citations(
        model="AnisotropicUNet",
        model_output="affinities" if is_aff_model else "boundaries"
    )
    cite["data"] = "https://doi.org/10.1101/2020.02.26.961037"

    doc = _get_doc(checkpoint, name, organelle)

    if additional_formats is None:
        additional_formats = []

    export_bioimageio_model(
        checkpoint, output,
        input_data=input_data,
        name=name,
        description=description,
        authors=[{"name": "Constantin Pape; @constantinpape"}],
        tags=tags,
        license="CC-BY-4.0",
        documentation=doc,
        git_repo="https://github.com/constantinpape/torch-em.git",
        cite=cite,
        model_postprocessing=postprocessing,
        input_optional_parameters=False,
        for_deepimagej="torchscript" in additional_formats,
        links=[get_bioimageio_dataset_id("platynereis")]
    )
    add_weight_formats(output, additional_formats)


if __name__ == "__main__":
    parser = export_parser_helper()
    args = parser.parse_args()
    export_to_bioimageio(args.checkpoint, args.output, args.input,
                         bool(args.affs_to_bd), args.additional_formats)
