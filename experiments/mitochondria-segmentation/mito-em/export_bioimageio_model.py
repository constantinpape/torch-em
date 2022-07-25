import os

from elf.io import open_file
from torch_em.data.datasets import get_bioimageio_dataset_id
from torch_em.util import (add_weight_formats, export_parser_helper,
                           export_bioimageio_model, get_default_citations,
                           get_training_summary)


def _get_name_and_description(is_aff):
    name = "MitochondriaEMSegmentation"
    if is_aff:
        name += "AffinityModel"
    else:
        name += "BoundaryModel"
    description = "Mitochondria segmentation for electron microscopy."
    return name, description


def _load_data(input_):
    with open_file(input_, 'r') as f:
        ds = f['raw']
        shape = ds.shape
        halo = [16, 128, 128]
        bb = tuple(slice(sh // 2 - ha, sh // 2 + ha) for sh, ha in zip(shape, halo))
        raw = ds[bb]
    return raw


def _get_doc(is_aff_model, ckpt, name):
    if is_aff_model:
        pred_type = "affinity maps"
        pp = "The affinities can be processed with the Mutex Watershed to obtain an instance segmentation."
    else:
        pred_type = "boundary maps"
        pp = "The boundaries can be processed with Multicut to obtain an instance segmentation."

    training_summary = get_training_summary(ckpt, to_md=True, lr=1.0e-4)
    model_tag = name.lower()
    doc = f"""# U-Net for Mitochondria Segmentation

This model segments mitochondria in electron microscopy images. It predicts {pred_type} and foreground probabilities. {pp}

## Training

The network was trained on data from the [MitoEM Segmentation Challenge](https://mitoem.grand-challenge.org/).
The training script can be found [here](https://github.com/constantinpape/torch-em/tree/main/experiments/mito-em).
This folder also includes example usages of this model.

### Training Data

- Imaging modality: serial blockface electron microscopy
- Dimensionality: 3D
- Source: https://mitoem.grand-challenge.org/

### Recommended Validation

It is recommended to validate the instance segmentation obtained from this model using intersection-over-union.
See [the validation script](https://github.com/constantinpape/torch-em/tree/main/experiments/mito-em/validate_model.py).
This model can also be used in ilastik, deepimageJ or other software that supports the bioimage.io model format.

### Training Schedule

{training_summary}

## Contact

For questions or issues with this models, please reach out by:
- opening a topic with tags bioimageio and {model_tag} on [image.sc](https://forum.image.sc/)
- or creating an issue in https://github.com/constantinpape/torch-em"""
    return doc


def export_to_bioimageio(checkpoint, input_, output, affs_to_bd, additional_formats):

    root, ckpt_name = os.path.split(checkpoint)
    if input_ is None:
        input_data = None
    else:
        input_data = _load_data(input_)

    is_aff_model = "affinity" in ckpt_name
    if is_aff_model and affs_to_bd:
        postprocessing = "affinities_with_foreground_to_boundaries3d"
    else:
        postprocessing = None

    if is_aff_model and affs_to_bd:
        is_aff_model = False

    name, desc = _get_name_and_description(is_aff_model)
    if is_aff_model:
        offsets = [
            [-1, 0, 0], [0, -1, 0], [0, 0, -1],
            [-2, 0, 0], [0, -3, 0], [0, 0, -3],
            [-3, 0, 0], [0, -9, 0], [0, 0, -9]
        ]
        config = {"mws": {"offsets": offsets}}
    else:
        config = {}

    cite = get_default_citations(
        model="AnisotropicUNet",
        model_output="affinities" if is_aff_model else "boundaries"
    )
    cite["data"] = "https://doi.org/10.1007/978-3-030-59722-1_7"
    tags = ["3d", "electron-microscopy", "mitochondria", "instance-segmentation", "unet"]

    doc = _get_doc(is_aff_model, checkpoint, name)

    if additional_formats is None:
        additional_formats = []

    export_bioimageio_model(
        checkpoint, output,
        input_data=input_data,
        name=name,
        description=desc,
        authors=[{"name": "Constantin Pape; @constantinpape"}],
        tags=tags,
        license="CC-BY-4.0",
        documentation=doc,
        git_repo="https://github.com/constantinpape/torch-em.git",
        cite=cite,
        model_postprocessing=postprocessing,
        input_optional_parameters=False,
        for_deepimagej="torchscript" in additional_formats,
        links=[get_bioimageio_dataset_id("mitoem")],
        maintainers=[{"github_user": "constantinpape"}],
        config=config,
    )
    add_weight_formats(output, additional_formats)


if __name__ == "__main__":
    parser = export_parser_helper()
    args = parser.parse_args()
    export_to_bioimageio(args.checkpoint, args.input, args.output,
                         bool(args.affs_to_bd), args.additional_formats)
