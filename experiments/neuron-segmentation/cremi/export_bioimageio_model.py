import os

from elf.io import open_file
from torch_em.data.datasets import get_bioimageio_dataset_id
from torch_em.util import (add_weight_formats, export_biomageio_model,
                           get_default_citations, export_parser_helper)


def _load_data(input_):
    with open_file(input_, 'r') as f:
        ds = f['volumes/raw']
        shape = ds.shape
        halo = [16, 180, 180]
        bb = tuple(slice(sh // 2 - ha, sh // 2 + ha) for sh, ha in zip(shape, halo))
        raw = ds[bb]
    return raw


def _get_name(is_aff):
    name = "CREMI"
    if is_aff:
        name += "-AffinityModel"
    else:
        name += "-BoundaryModel"
    return name


def _get_doc(is_aff_model):
    ndim = 3
    if is_aff_model:
        doc = f"""
## {ndim}D U-Net for Affinity Prediction

This model was trained on the data of the CREMI neuron segmentation challenge.
It predicts affinity maps that can be processed with the mutex watershed to obtain
an instance segmentation.
        """
    else:
        doc = f"""
## {ndim}D U-Net for Boundary Prediction

This model was trained on the data of the CREMI neuron segmentation challenge.
It predicts boundary maps that can be processed with multicut segmentation to obtain
an instance segmentation.
        """
    return doc


# TODO write offsets and other mws params into the config if this is a affinity model
def export_to_bioimageio(checkpoint, output, input_, affs_to_bd, additional_formats):

    ckpt_name = os.path.split(checkpoint)[1]

    if input_ is None:
        input_data = None
    else:
        input_data = _load_data(input_)

    is_aff_model = 'affinity' in ckpt_name
    if is_aff_model and affs_to_bd:
        postprocessing = 'affinities_to_boundaries_anisotropic'
    else:
        postprocessing = None

    if is_aff_model and affs_to_bd:
        is_aff_model = False
    name = _get_name(is_aff_model)
    tags = ["u-net", "neuron-segmentation", "segmentation", "volume-em", "cremi", "connectomics"]
    tags += ["boundary-prediction"] if is_aff_model else ["affinity-prediction"]

    # eventually we should refactor the citation logic
    cite = get_default_citations(
        model="AnisotropicUNet", model_output="affinities" if is_aff_model else "boundaries"
    )
    cite["data"] = "https://cremi.org"

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
        for_deepimagej="torchscript" in additional_formats,
        links=[get_bioimageio_dataset_id("cremi")]
    )
    add_weight_formats(output, additional_formats)


if __name__ == '__main__':
    parser = export_parser_helper()
    args = parser.parse_args()
    export_to_bioimageio(args.checkpoint, args.output, args.input,
                         bool(args.affs_to_bd), args.additional_formats)
