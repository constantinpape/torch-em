import os

from elf.io import open_file
from torch_em.data.datasets import get_bioimageio_dataset_id
from torch_em.util import (add_weight_formats, export_parser_helper,
                           export_biomageio_model, get_default_citations)


def _load_data(input_, organelle):
    if organelle == "cells":
        key = "volumes/raw/s1"
    else:
        key = "volumes/raw"
    with open_file(input_, 'r') as f:
        ds = f[key]
        shape = ds.shape
        halo = [16, 128, 128]
        bb = tuple(slice(sh // 2 - ha, sh // 2 + ha) for sh, ha in zip(shape, halo))
        raw = ds[bb]
    return raw


def _get_name(is_aff, organelle):
    name = f"Platyereis-{organelle}"
    if is_aff:
        name += "-AffinityModel"
    else:
        name += "-BoundaryModel"
    return name


def _get_doc(is_aff_model, organelle):
    ndim = 3
    if is_aff_model:
        doc = f"""
## {ndim}D U-Net for Affinity Prediction

This model was trained on data from a whole body EM volume of a Platynereis dumerilii larva.
It predicts affinity maps for {organelle} segmentation in SBEM volumes.
The affinities can be processed with the mutex watershed to obtain an instance segmentation."""
    else:
        doc = f"""
## {ndim}D U-Net for Boundary Prediction

This model was trained on data from a whole body EM volume of a Platynereis dumerilii larva.
It predicts boundary maps for {organelle} semgentation in SBEM volumes.
The boundaries can be processed with multicut segmentation to obtain an instance segmentation."""
    return doc


# TODO write offsets and other mws params into the config if this is a affinity model
def export_to_bioimageio(checkpoint, output, input_, affs_to_bd, additional_formats):

    root, ckpt_name = os.path.split(checkpoint)
    organelle = os.path.split(root)[0]
    assert organelle in ('cells', 'mitochondria', 'nuclei'), organelle

    if input_ is None:
        input_data = None
    else:
        input_data = _load_data(input_, organelle)

    is_aff_model = 'affinity' in ckpt_name
    if is_aff_model and affs_to_bd:
        if organelle in ('cells',):
            postprocessing = 'affinities_to_boundaries_anisotropic'
        elif organelle in ('mitochondria', 'nuclei'):
            postprocessing = 'affinities_with_foreground_to_boundaries3d'
    else:
        postprocessing = None

    if is_aff_model and affs_to_bd:
        is_aff_model = False
    name = _get_name(is_aff_model, organelle)
    tags = ["u-net", f"{organelle}-segmentation", "segmentation", "volume-em", "platynereis", organelle]
    tags += ["boundary-prediction"] if is_aff_model else ["affinity-prediction"]

    cite = get_default_citations(
        model="AnisotropicUNet",
        model_output="affinities" if is_aff_model else "boundaries"
    )
    cite["data"] = "https://doi.org/10.1101/2020.02.26.961037"

    doc = _get_doc(is_aff_model, organelle)

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
        for_deepimagej="torchscript" in additional_formats,
        links=[get_bioimageio_dataset_id("platynereis")]
    )
    add_weight_formats(output, additional_formats)


if __name__ == '__main__':
    parser = export_parser_helper()
    args = parser.parse_args()
    export_to_bioimageio(args.checkpoint, args.output, args.input,
                         bool(args.affs_to_bd), args.additional_formats)
