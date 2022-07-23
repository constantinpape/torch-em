import argparse
import os
from glob import glob

import numpy as np
from elf.io import open_file
from torch_em.data.datasets import get_bioimageio_dataset_id
from torch_em.util import (add_weight_formats,
                           export_bioimageio_model,
                           get_default_citations,
                           get_shallow2deep_config,
                           get_training_summary)
from torch_em.shallow2deep.shallow2deep_model import RFWithFilters, _get_filters


def _get_name_and_description(is3d, name):
    if name is None:
        name = "EnhancerMitochondriaEM3D" if is3d else "EnhancerMitochondriaEM2D"
    description = "Prediction enhancer for segmenting mitochondria in EM images."
    return name, description


def _get_doc(ckpt, name, is3d):
    training_summary = get_training_summary(ckpt, to_md=True, lr=1.0e-4)
    doc = f"""#Prediction Enhancer for Mitochondrion Segmentation in EM

This model was trained with the [Shallow2Deep](https://doi.org/10.3389/fcomp.2022.805166)
method to improve ilastik predictions for mitochondria segmentation in EM.
It predicts foreground and boundary probabilities.

## Training

The network was trained on data from [MitoEM](https://doi.org/10.1007/978-3-030-59722-1_7)
and trained using [torch_em](https://github.com/constantinpape/torch-em).

### Training Data

- Imaging modality: Electron Microscopy
- Dimensionality: f{"3D" if is3d else "2D"}
- Source: https://doi.org/10.1007/978-3-030-59722-1_7

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


def create_input_2d(input_path, rf_path):
    with open_file(input_path, "r") as f:
        data = f["raw"][-1, :512, :512]
    assert os.path.exists(rf_path), rf_path
    filter_config = _get_filters(2, None)
    rf = RFWithFilters(rf_path, ndim=2, filter_config=filter_config, output_channel=1)
    pred = rf(data)
    return pred[None]


def create_input_anisotropic(input_path, rf_path):
    with open_file(input_path, "r") as f:
        data = f["raw"][:32, :256, :256]
    assert os.path.exists(rf_path), rf_path
    filter_config = _get_filters(2, None)
    rf = RFWithFilters(rf_path, ndim=2, filter_config=filter_config, output_channel=1)
    pred = np.zeros(data.shape, dtype="float32")
    for z in range(data.shape[0]):
        pred[z] = rf(data[z])
    return pred[None]


def create_input_3d(input_path, rf_path):
    with open_file(input_path, "r") as f:
        data = f["raw"][:32, :256, :256]
    assert os.path.exists(rf_path), rf_path
    filter_config = _get_filters(3, None)
    rf = RFWithFilters(rf_path, ndim=3, filter_config=filter_config, output_channel=1)
    pred = rf(data)
    return pred[None]


def export_enhancer(input_, is3d, checkpoint=None, version=None, name=None):

    if checkpoint is None:
        checkpoint = "./checkpoints/shallow2deep-mitoem3d" if is3d else\
            "./checkpoints/shallow2deep-mitoem2d"
        out_folder = "./bio-models"
    else:
        assert version is not None
        out_folder = f"./bio-models/v{version}"

    if is3d == "anisotropic":
        rf_path = "/scratch/pape/s2d-mitochondria/rfs2d-worst_points/mitoem/rf_0499.pkl"
        input_data = create_input_anisotropic(input_, rf_path)
        is3d = True
    elif is3d:
        assert False, "Currently don't have 3d rfs for mitos"
        input_data = create_input_3d(input_, rf_path)
    else:
        rf_path = "/scratch/pape/s2d-mitochondria/rfs2d-worst_points/mitoem/rf_0499.pkl"
        input_data = create_input_2d(input_, rf_path)

    name, description = _get_name_and_description(is3d, name)
    tags = ["unet", "mitochondria", "electron-microscopy", "instance-segmentation", "shallow2deep"]
    tags += ["3d"] if is3d else ["2d"]

    # eventually we should refactor the citation logic
    mitoem_doi = "https://doi.org/10.1007/978-3-030-59722-1_7"
    s2d_doi = "https://doi.org/10.3389/fcomp.2022.805166"
    cite = get_default_citations(model="UNet3d" if is3d else "UNet2d", model_output="boundaries")
    cite.append({"text": "data", "doi": mitoem_doi})
    cite.append({"text": "shallow2deep", "doi": s2d_doi})
    doc = _get_doc(checkpoint, name, is3d)
    additional_formats = ["torchscript"]

    os.makedirs(out_folder, exist_ok=True)
    output = os.path.join(out_folder, name)

    if is3d:
        min_shape = [16, 128, 128]
        halo = [4, 32, 32]
    else:
        min_shape = [256, 256]
        halo = [32, 32]

    config = get_shallow2deep_config(rf_path)
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
        training_data={"id": get_bioimageio_dataset_id("mitoem")},
        maintainers=[{"github_user": "constantinpape"}],
        min_shape=min_shape,
        halo=halo,
        config=config,
    )
    add_weight_formats(output, additional_formats)


def export_version(args):

    def _get_ndim(x):
        if x == "2d":
            return False
        elif x == "anisotropic":
            return x
        elif x == "3d":
            return True
        raise ValueError(x)

    checkpoints = glob("./checkpoints/s2d-em-mitos-*")
    out_folder = f"./bio-models/v{args.version}"
    for ckpt in checkpoints:
        name = os.path.basename(ckpt)
        if(os.path.exists(os.path.join(out_folder, name))):
            continue
        parts = name.split("-")
        is3d = _get_ndim(parts[-2])
        assert is3d is not None
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("Exporting:", ckpt)
        print(name)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        export_enhancer(args.input, is3d, checkpoint=ckpt, version=args.version, name=name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-d", "--is3d", default=0)
    parser.add_argument("-v", "--version", default=None, type=int)
    args = parser.parse_args()
    if args.version is None:
        export_enhancer(args.input, args.is3d)
    else:
        # export all currently trained checkpoints as one version
        export_version(args)


if __name__ == "__main__":
    main()
