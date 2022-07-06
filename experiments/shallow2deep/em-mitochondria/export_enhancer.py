import argparse
import os
from glob import glob

from elf.io import open_file
from torch_em.data.datasets import get_bioimageio_dataset_id
from torch_em.util import (add_weight_formats,
                           export_bioimageio_model,
                           get_default_citations,
                           get_training_summary)
from torch_em.shallow2deep.shallow2deep_model import RFWithFilters, _get_filters


def _get_name_and_description():
    name = "EnhancerMitochondriaEM2D"
    description = "Prediction enhancer for segmenting mitochondria in EM images."
    return name, description


def _get_doc(ckpt, name):
    training_summary = get_training_summary(ckpt, to_md=True, lr=1.0e-4)
    doc = f"""#Prediction Enhancer for Mitochondrion Segmentation in EM

This model was trained with the [Shallow2Deep](https://doi.org/10.3389/fcomp.2022.805166)
method to improve ilastik predictions for mitochondria segmentation in EM.
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
Note that this model expects foreground probabilities from a shallow classifer,
such as a Random Forest, as input. It can thus be applied to new data of mitochondria, where only a
new Random Forest for mitochondria foreground prediction needs to be trained, e.g. using ilastik.
This model can be used in ilastik, deepimageJ or other software that supports the bioimage.io model format.

### Training Schedule

{training_summary}

## Contact

For questions or issues with this models, please reach out by:
- opening a topic with tags bioimageio and the name of this model on [image.sc](https://forum.image.sc/)
- or creating an issue in https://github.com/constantinpape/torch-em"""
    return doc


def create_input(input_path, checkpoint):
    with open_file(input_path, "r") as f:
        data = f["raw"][-1, :512, :512]
    rf_path = glob(os.path.join(checkpoint, "rfs/*.pkl"))[-1]
    assert os.path.exists(rf_path), rf_path
    filter_config = _get_filters(2, None)
    rf = RFWithFilters(rf_path, ndim=2, filter_config=filter_config, output_channel=1)
    pred = rf(data)
    return pred[None]


def export_enhancer(input_, train_advanced):

    checkpoint = "./checkpoints/shallow2deep-em-mitochondria"
    if train_advanced:
        checkpoint += "-advanced"
    input_data = create_input(input_, checkpoint)

    name, description = _get_name_and_description()
    tags = ["unet", "mitochondria", "electron-microscopy", "instance-segmentation", "2d", "shallow2deep"]

    # eventually we should refactor the citation logic
    vnc_doi = "http://dx.doi.org/10.6084/m9.figshare.856713"
    s2d_doi = "https://doi.org/10.3389/fcomp.2022.805166"
    cite = get_default_citations(model="UNet2d", model_output="boundaries")
    cite.append({"text": "data", "doi": vnc_doi})
    cite.append({"text": "shallow2deep", "doi": s2d_doi})
    doc = _get_doc(checkpoint, name)
    additional_formats = ["torchscript"]

    out_folder = "./bio-models"
    os.makedirs(out_folder, exist_ok=True)
    output = os.path.join(out_folder, f"{name}-advanced-traing" if train_advanced else name)

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
    )
    add_weight_formats(output, additional_formats)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input")
    parser.add_argument("-a", "--train_advanced", default=0)
    args = parser.parse_args()
    export_enhancer(args.input, args.train_advanced)


if __name__ == "__main__":
    main()
