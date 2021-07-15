import os

import h5py
from torch_em.data.datasets import get_bioimageio_dataset_id
from torch_em.util import (add_weight_formats, export_biomageio_model,
                           get_default_citations, export_parser_helper)


def _get_name(is_aff):
    name = "Covid-IF-Cells"
    if is_aff:
        name += "-AffinityModel"
    else:
        name += "-BoundaryModel"
    return name


def _get_doc(is_aff_model):
    training_url = "https://github.com/constantinpape/torch-em/tree/main/experiments/neuron-segmentation/covid-if"
    if is_aff_model:
        title = "U-Net for Affinity Prediction"
        output_name = "affinity maps"
        seg = "with the mutex watershed "
    else:
        title = "U-Net for Boundary Prediction"
        output_name = "boundary maps"
        seg = "by seeded watershed segmentation"

    doc = f"""
## {title}

This model was trained on HTM data from a immunofluorescence based Covid-19 antibody test
for cell segmentation.
It predicts {output_name} that can be processed {seg} to obtain an instance segmentation.
For more details, check out [the training scripts]({training_url})."""

    return doc


# TODO write offsets and other mws params into the config if this is a affinity model
def export_to_bioimageio(checkpoint, output, input_, affs_to_bd, additional_formats):

    ckpt_name = os.path.split(checkpoint)[1]
    if input_ is None:
        input_data = None
    else:
        with h5py.File(input_, 'r') as f:
            input_data = f['raw/serum_IgG/s0'][:]

    is_aff_model = 'affinity' in ckpt_name
    if is_aff_model and affs_to_bd:
        postprocessing = 'affinities_with_foreground_to_boundaries2d'
    else:
        postprocessing = None
    if is_aff_model and affs_to_bd:
        is_aff_model = False

    name = _get_name(is_aff_model)
    tags = ["u-net", "cell-segmentation", "htm", "high-throughput-microscopt", "segmentation", "cells",
            "covid-antibody-test", "covid-19", "sars-cov-2", "immunofluorescence"]
    tags += ["boundary-prediction"] if is_aff_model else ["affinity-prediction"]

    # eventually we should refactor the citation logic
    covid_if_pub = "https://doi.org/10.1002/bies.202000257"
    cite = get_default_citations(
        model="UNet2d", model_output="affinities" if is_aff_model else "boundaries"
    )
    cite["data"] = covid_if_pub
    if not is_aff_model:
        cite["segmentation algorithm"] = covid_if_pub

    doc = _get_doc(is_aff_model)

    export_biomageio_model(
        checkpoint, output,
        input_data=input_data,
        name=name,
        authors=[{"name": "Constantin Pape; @constantinpape"}],
        tags=tags,
        license='CC-BY-4.0',
        documentation=doc,
        git_repo='https://github.com/constantinpape/torch-em.git',
        cite=cite,
        model_postprocessing=postprocessing,
        input_optional_parameters=False,
        # need custom deepimagej fields if we have torchscript export
        for_deepimagej="torchscript" in additional_formats,
        links=[get_bioimageio_dataset_id("covid_if")]
    )
    add_weight_formats(output, additional_formats)


if __name__ == '__main__':
    parser = export_parser_helper()
    args = parser.parse_args()
    export_to_bioimageio(args.checkpoint, args.output, args.input,
                         bool(args.affs_to_bd), args.additional_formats)
