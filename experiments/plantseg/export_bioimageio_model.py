import os

from elf.io import open_file
from torch_em.data.datasets import get_bioimageio_dataset_id
from torch_em.util import (add_weight_formats, export_biomageio_model,
                           get_default_citations, export_parser_helper)


def _load_data(input_, is2d):
    with open_file(input_, 'r') as f:
        ds = f['raw']
        shape = ds.shape
        if is2d:
            halo = [256, 256]
            bb = (shape[0] // 2,) + tuple(slice(sh // 2 - ha, sh // 2 + ha) for sh, ha in zip(shape[1:], halo))
        else:
            halo = [16, 128, 128]
            bb = tuple(slice(sh // 2 - ha, sh // 2 + ha) for sh, ha in zip(shape, halo))
        raw = ds[bb]
    return raw


def _get_name(is_aff, specimen):
    name = f"Arabidopsis-{specimen}"
    if is_aff:
        name += "-AffinityModel"
    else:
        name += "-BoundaryModel"
    return name


def _get_doc(is_aff_model, specimen):
    ndim = 3
    if is_aff_model:
        doc = f"""
## {ndim}D U-Net for Affinity Prediction

This model was trained on data from light microscopy data of Arabidopsis thaliana {specimen}.
It predicts affinity maps for {specimen} segmentation.
The affinities can be processed with the mutex watershed to obtain an instance segmentation.
        """
    else:
        doc = f"""
## {ndim}D U-Net for Boundary Prediction

This model was trained on data from light microscopy data of Arabidopsis thaliana {specimen}.
It predicts boundary maps for {specimen} segmentation.
The boundaries can be processed with multicut segmentation to obtain an instance segmentation.
        """
    return doc


# TODO write offsets and other mws params into the config if this is a affinity model
def export_to_bioimageio(checkpoint, output, input_, affs_to_bd, additional_formats):

    root, ckpt_name = os.path.split(checkpoint)
    specimen = os.path.split(root)[0]
    assert specimen in ("ovules", "roots"), specimen

    is2d = checkpoint.endswith('2d')
    if input_ is None:
        input_data = None
    else:
        input_data = _load_data(input_, is2d)

    is_aff_model = "affinity" in ckpt_name
    if is_aff_model and affs_to_bd:
        postprocessing = "affinities_to_boundaries3d"
    else:
        postprocessing = None

    if is_aff_model and affs_to_bd:
        is_aff_model = False
    name = _get_name(is_aff_model, specimen)
    tags = ["u-net", f"{specimen}-segmentation", "segmentation", "light-microscopy", "arabidopsis"]
    if specimen == "ovules":
        tags += ["ovules", "confocal-microscopy"]
    else:
        tags += ["primordial-root", "light-sheet-microscopy"]
    tags += ["boundary-prediction"] if is_aff_model else ["affinity-prediction"]

    # eventually we should refactor the citation logic
    plantseg_pub = "https://doi.org/10.7554/eLife.57613.sa2"
    cite = get_default_citations(
        model="UNet2d" if is2d else "UNet3d", model_output="affinities" if is_aff_model else "boundaries"
    )
    cite["data"] = plantseg_pub
    cite["segmentation algorithm"] = plantseg_pub

    doc = _get_doc(is_aff_model, specimen)

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
        links=[get_bioimageio_dataset_id("ovules")]
    )
    add_weight_formats(output, additional_formats)


if __name__ == '__main__':
    parser = export_parser_helper()
    args = parser.parse_args()
    export_to_bioimageio(args.checkpoint, args.output, args.input,
                         bool(args.affs_to_bd), args.additional_formats)
