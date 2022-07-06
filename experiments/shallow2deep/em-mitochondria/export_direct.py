import argparse
import os

import h5py
from torch_em.data.datasets import get_bioimageio_dataset_id
from torch_em.util import (add_weight_formats,
                           export_bioimageio_model,
                           get_default_citations,
                           get_training_summary)


def _get_name_and_description():
    name = "MitchondriaEMSegmentation2D"
    description = "Segmentation of mitochondria in EM images."
    return name, description


def _get_doc(ckpt, name):
    training_summary = get_training_summary(ckpt, to_md=True, lr=1.0e-4)
    model_tag = name.lower()
    doc = f"""#Prediction Enhancer for Mitochondrion Segmentation in EM

This model was trained to segment mitochondria in EM.
It predicts foreground and boundary probabilities.

## Training

The network was trained on data from [the VNC dataset](http://dx.doi.org/10.6084/m9.figshare.856713)
and trained using [torch_em](https://github.com/constantinpape/torch-em).

### Training Data

- Imaging modality: Electron Microscopy
- Dimensionality: 2D
- Source: http://dx.doi.org/10.6084/m9.figshare.856713

### Recommended Validation

It is recommended to validate the instance segmentation obtained from this model using intersection-over-union.
This model can be used in ilastik, deepimageJ or other software that supports the bioimage.io model format.

### Training Schedule

{training_summary}

## Contact

For questions or issues with this models, please reach out by:
- opening a topic with tags bioimageio and {model_tag} on [image.sc](https://forum.image.sc/)
- or creating an issue in https://github.com/constantinpape/torch-em"""
    return doc


def create_input(input_, checkpoint):
    input_path = os.path.join(input_, "vnc_train.h5")
    assert os.path.exists(input_path), input_path
    with h5py.File(input_path, "r") as f:
        data = f["raw"][-1, :512, :512]
    return data[None, None]


def export_enhancer(checkpoint, input_):
    input_data = create_input(input_, checkpoint)
    name, description = _get_name_and_description()
    tags = ["unet", "mitochondria", "electron-microscopy", "instance-segmentation", "2d"]

    # eventually we should refactor the citation logic
    vnc_doi = "http://dx.doi.org/10.6084/m9.figshare.856713"
    cite = get_default_citations(model="UNet2d", model_output="boundaries")
    cite.append({"text": "data", "doi": vnc_doi})
    doc = _get_doc(checkpoint, name)
    additional_formats = ["torchscript"]

    out_folder = "./bio-models"
    os.makedirs(out_folder, exist_ok=True)
    output = os.path.join(out_folder, name)

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
        input_optional_parameters=False,
        # need custom deepimagej fields if we have torchscript export
        for_deepimagej="torchscript" in additional_formats,
        training_data={"id": get_bioimageio_dataset_id("vnc")},
        maintainers=[{"github_user": "constantinpape"}],
        min_shape=[256, 256],
    )
    add_weight_formats(output, additional_formats)


def main():
    parser = argparse.ArgumentParser()
    # checkpoint = "./checkpoints/em-mitochondria"
    parser.add_argument("-c", "--checkpoint")
    parser.add_argument("-i", "--input")
    args = parser.parse_args()
    export_enhancer(args.checkpoint, args.input)


if __name__ == "__main__":
    main()
