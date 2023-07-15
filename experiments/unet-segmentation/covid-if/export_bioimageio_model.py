import os

import h5py
import torch_em
from torch_em.data.datasets import get_bioimageio_dataset_id
from torch_em.util.modelzoo import (add_weight_formats,
                                    export_bioimageio_model,
                                    export_parser_helper,
                                    get_default_citations)


def _get_name_and_description(is_aff):
    name = "CovidIFCellSegmentation"
    if is_aff:
        name += "AffinityModel"
    else:
        name += "BoundaryModel"
    description = "Cell segmentation for immunofluorescence microscopy."
    return name, description


def _get_doc(is_aff_model, ckpt, name):
    if is_aff_model:
        pred_type = "affinity maps"
        pp = "The affinities can be processed with the Mutex Watershed to obtain an instance segmentation."
    else:
        pred_type = "boundary maps"
        pp = ("The boundaries can be combined with a nucleus segmentation to obtain an instance segmentation "
              "via seeded watershed.")

    training_summary = torch_em.util.get_training_summary(ckpt, to_md=True, lr=1.0e-4)
    model_tag = name.lower()
    doc = f"""# U-Net for Cell Segmentation in IF

This model segments cells in immunofluorescence microscopy images.
It predicts {pred_type} and foreground probabilities and was trained on Vero E6 cells
imaged with a high-throughput-microscope, as part of a Covid19 antibody test.
{pp}

## Training

The network was trained on data from [this publication](https://doi.org/10.1002/bies.202000257).
The training script can be found [here](https://github.com/constantinpape/torch-em/tree/main/experiments/covid-if).
This folder also includes example usages of this model.

### Training Data

- Imaging modality: immunofluorescence images of Vero E6 cells
- Dimensionality: 2D
- Source: https://zenodo.org/record/5092850

### Recommended Validation

It is recommended to validate the instance segmentation obtained from this model using intersection-over-union.
See [the validation script](https://github.com/constantinpape/torch-em/tree/main/experiments/covid-if/validate_model.py).
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
        with h5py.File(input_, "r") as f:
            input_data = f["raw/serum_IgG/s0"][:512, :512]

    is_aff_model = "affinity" in ckpt_name
    if is_aff_model and affs_to_bd:
        postprocessing = "affinities_with_foreground_to_boundaries2d"
    else:
        postprocessing = None
    if is_aff_model and affs_to_bd:
        is_aff_model = False

    name, description = _get_name_and_description(is_aff_model)
    tags = ["unet", "cells", "high-content-microscopy", "instance-segmentation",
            "covid19", "immunofluorescence-microscopy", "2d"]

    # eventually we should refactor the citation logic
    covid_if_pub = "https://doi.org/10.1002/bies.202000257"
    cite = get_default_citations(
        model="UNet2d", model_output="affinities" if is_aff_model else "boundaries"
    )
    cite["data"] = covid_if_pub
    if not is_aff_model:
        cite["segmentation algorithm"] = covid_if_pub

    doc = _get_doc(is_aff_model, checkpoint, name)
    if is_aff_model:
        offsets = [
            [-1, 0], [0, -1],
            [-3, 0], [0, -3],
            [-9, 0], [0, -9],
            [-27, 0], [0, -27]
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
        authors=[{"name": "Constantin Pape", "affiliation": "EMBL Heidelberg"}],
        tags=tags,
        license="CC-BY-4.0",
        documentation=doc,
        description=description,
        git_repo="https://github.com/constantinpape/torch-em.git",
        cite=cite,
        config=config,
        model_postprocessing=postprocessing,
        input_optional_parameters=False,
        # need custom deepimagej fields if we have torchscript export
        for_deepimagej="torchscript" in additional_formats,
        links=[get_bioimageio_dataset_id("covid_if")],
        maintainers=[{"github_user": "constantinpape"}],
    )
    add_weight_formats(output, additional_formats)


if __name__ == "__main__":
    parser = export_parser_helper()
    args = parser.parse_args()
    export_to_bioimageio(args.checkpoint, args.output, args.input,
                         bool(args.affs_to_bd), args.additional_formats)
