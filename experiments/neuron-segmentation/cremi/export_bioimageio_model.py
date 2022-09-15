import os

import torch_em
from elf.io import open_file
from torch_em.data.datasets import get_bioimageio_dataset_id
from torch_em.util.modelzoo import (add_weight_formats, export_bioimageio_model,
                                    get_default_citations, export_parser_helper)


def _load_data(input_):
    with open_file(input_, "r") as f:
        ds = f["volumes/raw"]
        shape = ds.shape
        halo = [16, 180, 180]
        bb = tuple(slice(sh // 2 - ha, sh // 2 + ha) for sh, ha in zip(shape, halo))
        raw = ds[bb]
    return raw


def _get_name_and_description(is_aff):
    name = "NeuronEMSegmentation"
    if is_aff:
        name += "AffinityModel"
    else:
        name += "BoundaryModel"
    description = "Neuron segmentation in EM, trained on the CREMI challenge data."
    return name, description


def _get_doc(is_aff_model, ckpt, name):
    if is_aff_model:
        pred_type = "affinity maps"
        pp = "The affinities can be processed with the Mutex Watershed to obtain an instance segmentation."
    else:
        pred_type = "boundary maps"
        pp = "The boundaries can be processed with Multicut segmentation to obtain an instance segmentation."

    training_summary = torch_em.util.get_training_summary(ckpt, to_md=True, lr=1.0e-4)
    model_tag = name.lower()
    doc = f"""# U-Net for EM Neuron Segmentation

This model segments neurons in electron microscopy images. It predicts {pred_type} segmenting neural membranes.
{pp}

## Training

The network was trained on data from the CREMI Neuron Segmentation Challenge.
The training script can be found [here](https://github.com/constantinpape/torch-em/tree/main/experiments/neuron_segmentation/cremi).
This folder also includes example usages of this model.

### Training Data

- Imaging modality: serial section transmission electron microscopy data of fruit-fly neural tissue.
- Dimensionality: 3D
- Source: CREMI Neuron Segmentation Challenge

### Recommended Validation

It is recommended to validate the instance segmentation obtained from this model using metrics derived from the RandIndex
or the Variation of Information.
See [the validation script](https://github.com/constantinpape/torch-em/tree/main/experiments/neuron_segmentation/cremi/validate_model.py).
This model can also be used in ilastik, deepimageJ or other software that supports the bioimage.io model format.

### Training Schedule

{training_summary}

## Contact

For questions or issues with this models, please reach out by:
- opening a topic with tags bioimageio and {model_tag} on [image.sc](https://forum.image.sc/)
- or creating an issue in https://github.com/constantinpape/torch-em"""
    return doc


def export_to_bioimageio(checkpoint, output, input_, affs_to_bd, additional_formats):

    ckpt_name = os.path.split(checkpoint)[1]

    if input_ is None:
        input_data = None
    else:
        input_data = _load_data(input_)

    is_aff_model = "affinity" in ckpt_name
    if is_aff_model and affs_to_bd:
        postprocessing = "affinities_to_boundaries_anisotropic"
    else:
        postprocessing = None

    if is_aff_model and affs_to_bd:
        is_aff_model = False
    name, description = _get_name_and_description(is_aff_model)
    tags = ["unet", "neurons", "instance-segmentation", "electron-microscopy", "cremi", "connectomics", "3d"]
    tags += ["boundary-prediction"] if is_aff_model else ["affinity-prediction"]

    # eventually we should refactor the citation logic
    cite = get_default_citations(
        model="AnisotropicUNet", model_output="affinities" if is_aff_model else "boundaries"
    )
    cite["data"] = "https://cremi.org"

    doc = _get_doc(is_aff_model, checkpoint, name)
    if is_aff_model:
        offsets = [
            [-1, 0, 0], [0, -1, 0], [0, 0, -1],
            [-2, 0, 0], [0, -3, 0], [0, 0, -3],
            [-3, 0, 0], [0, -9, 0], [0, 0, -9],
            [-4, 0, 0], [0, -27, 0], [0, 0, -27]
        ]
        config = {"mws": {"offsets": offsets}}
    else:
        config = {}

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
        links=[get_bioimageio_dataset_id("cremi")],
        config=config
    )
    if additional_formats:
        add_weight_formats(output, additional_formats)


if __name__ == "__main__":
    parser = export_parser_helper()
    args = parser.parse_args()
    export_to_bioimageio(args.checkpoint, args.output, args.input,
                         bool(args.affs_to_bd), args.additional_formats)
