import os

from elf.io import open_file
from torch_em.data.datasets import get_bioimageio_dataset_id
from torch_em.util import (add_weight_formats, export_parser_helper,
                           export_biomageio_model, get_default_citations)


def _load_data(input_):
    with open_file(input_, 'r') as f:
        ds = f['raw']
        shape = ds.shape
        halo = [16, 128, 128]
        bb = tuple(slice(sh // 2 - ha, sh // 2 + ha) for sh, ha in zip(shape, halo))
        raw = ds[bb]
    return raw


def _get_doc(is_aff_model):
    training_url = "https://github.com/constantinpape/torch-em/tree/main/experiments/neuron-segmentation/mito-em"
    if is_aff_model:
        title = "3D U-Net for Affinity Prediction"
        output_name = "affinity maps"
        seg = "with the mutex watershed "
    else:
        title = "3D U-Net for Boundary Prediction"
        output_name = "boundary maps"
        seg = "multicut segmentation"

    doc = f"""
## {title}

This model was trained on the data of the MitoEM dataset to perform mitochondria segmentaiton in EM.
It predicts {output_name} that can be processed {seg} to obtain an instance segmentation.
For more details, check out [the training scripts]({training_url})."""
    return doc


# TODO write offsets and other mws params into the config if this is a affinity model
def export_to_bioimageio(checkpoint, input_, output, affs_to_bd, additional_formats):

    root, ckpt_name = os.path.split(checkpoint)
    if input_ is None:
        input_data = None
    else:
        input_data = _load_data(input_)

    is_aff_model = 'affinity' in ckpt_name
    if is_aff_model and affs_to_bd:
        postprocessing = 'affinities_with_foreground_to_boundaries3d'
    else:
        postprocessing = None

    if is_aff_model and affs_to_bd:
        is_aff_model = False

    if is_aff_model:
        name = "EM-Mitochondria-AffinityModel"
    else:
        name = "EM-Mitochondria-BoundaryModel"

    tags = ["u-net", "mitochondria-segmentation",
            "segmentation", "mito-em", "mitochondria"]
    tags += ["boundary-prediction"] if is_aff_model else ["affinity-prediction"]

    # eventually we should refactor the citation logic
    cite = get_default_citations(
        model="AnisotropicUNet",
        model_output="affinities" if is_aff_model else "boundaries"
    )
    cite["data"] = "https://doi.org/10.1007/978-3-030-59722-1_7"

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
        links=[get_bioimageio_dataset_id("mitoem")]
    )
    add_weight_formats(output, additional_formats)


if __name__ == '__main__':
    parser = export_parser_helper()
    args = parser.parse_args()
    export_to_bioimageio(args.checkpoint, args.input, args.output,
                         bool(args.affs_to_bd), args.additional_formats)
