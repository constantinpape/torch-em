import torch_em
from elf.io import open_file
# from torch_em.data.datasets import get_bioimageio_dataset_id
from torch_em.util.modelzoo import (add_weight_formats, export_bioimageio_model,
                                    get_default_citations, export_parser_helper)


def _load_data(input_):
    with open_file(input_, "r") as f:
        raw = f["raw"][:1024, :1024]
    return raw


def _get_name_and_description():
    name = "AxonEMSegmentationModel"
    description = "Axon and myelin segmentation in EM, trained on the AxonDeepSeg data."
    return name, description


def _get_doc(ckpt, name):
    training_summary = torch_em.util.get_training_summary(ckpt, to_md=True, lr=1.0e-4)
    model_tag = name.lower()
    url = "https://github.com/constantinpape/torch-em/tree/main/experiments/neuron_segmentation/axondeepseg"
    doc = f"""# U-Net for Axon and Myelin EM Segmentation

This model segments axon and myelin in electron microscopy images.

## Training

The network was trained on data from the AxonDeepSeg Publication.
The training script can be found [here]({url}).
This folder also includes example usages of this model.

### Training Data

- Imaging modality: SEM and TEM data of mammalian neural tissue.
- Dimensionality: 2D
- Source: AxonDeepSeg

### Recommended Validation

It is recommended to validate the instance segmentation obtained from this model using F1-score
or Dice score against manual axon segmentations.
This model can also be used in ilastik, deepimageJ or other software that supports the bioimage.io model format.

### Training Schedule

{training_summary}

## Contact

For questions or issues with this models, please reach out by:
- opening a topic with tags bioimageio and {model_tag} on [image.sc](https://forum.image.sc/)
- or creating an issue in https://github.com/constantinpape/torch-em"""
    return doc


def export_to_bioimageio(checkpoint, output, input_, additional_formats):

    if input_ is None:
        input_data = None
    else:
        input_data = _load_data(input_)

    name, description = _get_name_and_description()
    tags = ["unet", "neurons", "semantic-segmentation", "electron-microscopy", "2d"]

    # eventually we should refactor the citation logic
    cite = get_default_citations(model="UNet2d")
    # cite["data"] = "https://doi.org/10.1038/s41598-018-22181-4"

    if additional_formats is None:
        additional_formats = []

    doc = _get_doc(checkpoint, name)
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
        input_optional_parameters=False,
        for_deepimagej="torchscript" in additional_formats,
        # TODO
        # links=[get_bioimageio_dataset_id("axondeepseg")],
    )
    if additional_formats:
        add_weight_formats(output, additional_formats)


if __name__ == "__main__":
    parser = export_parser_helper()
    args = parser.parse_args()
    export_to_bioimageio(args.checkpoint, args.output, args.input, args.additional_formats)
