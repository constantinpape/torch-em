import argparse
import os
from glob import glob

from elf.io import open_file
# from torch_em.data.datasets import get_bioimageio_dataset_id
from torch_em.util import (add_weight_formats,
                           export_bioimageio_model,
                           get_default_citations,
                           get_training_summary)
from torch_em.shallow2deep.shallow2deep_model import RFWithFilters, _get_filters


def _get_doc(ckpt, name, is3d):
    training_summary = get_training_summary(ckpt, to_md=True, lr=1.0e-4)
    doc = f"""#Prediction Enhancer for Membrane Segmentation in LM

This model was trained with the [Shallow2Deep](https://doi.org/10.3389/fcomp.2022.805166)
method to improve ilastik predictions for mitochondria segmentation in EM.
It predicts foreground and boundary probabilities.

## Training

The network was trained on light microscopy data of membranes
and trained using [torch_em](https://github.com/constantinpape/torch-em).

### Training Data

- Imaging modality: Light Microscopy
- Dimensionality: f{"3D" if is3d else "2D"}
- Source:

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


def create_input_2d():
    input_path = "/scratch/pape/s2d-lm-boundaries/mouse-embryo/Membrane/train/ginst_membrane_filt_E2_last_predictions_gasp_average_raw_corrected.h5"
    with open_file(input_path, "r") as f:
        data = f["raw"][-1, :512, :512]
    rf_path = "/scratch/pape/s2d-lm-boundaries/rfs2d-worst_tiles/mouse-embryo/rf_0499.pkl"
    filter_config = _get_filters(2, None)
    rf = RFWithFilters(rf_path, ndim=2, filter_config=filter_config, output_channel=1)
    pred = rf(data)
    return pred[None]


def create_input_3d():
    input_path = "/scratch/pape/s2d-lm-boundaries/mouse-embryo/Membrane/train/ginst_membrane_filt_E2_last_predictions_gasp_average_raw_corrected.h5"
    with open_file(input_path, "r") as f:
        data = f["raw"][:32, :256, :256]
    rf_path = "/scratch/pape/s2d-lm-boundaries/rfs3d-worst_tiles/mouse-embryo/rf_0499.pkl"
    filter_config = _get_filters(2, None)
    rf = RFWithFilters(rf_path, ndim=3, filter_config=filter_config, output_channel=1)
    pred = rf(data)
    return pred[None]


def export_enhancer(is3d, checkpoint, version, name):

    input_data = create_input_3d() if is3d else create_input_2d()

    description = "Prediction enhancer for segmenting membranes in light microscopy."
    tags = ["unet", "mitochondria", "electron-microscopy", "instance-segmentation", "shallow2deep"]
    tags += ["3d"] if is3d else ["2d"]

    # eventually we should refactor the citation logic
    # mitoem_doi = "https://doi.org/10.1007/978-3-030-59722-1_7"
    s2d_doi = "https://doi.org/10.3389/fcomp.2022.805166"
    cite = get_default_citations(model="UNet3d" if is3d else "UNet2d", model_output="boundaries")
    # cite.append({"text": "data", "doi": mitoem_doi})
    cite.append({"text": "shallow2deep", "doi": s2d_doi})
    doc = _get_doc(checkpoint, name, is3d)
    additional_formats = ["torchscript"]

    out_folder = f"./bio-models/v{version}"
    os.makedirs(out_folder, exist_ok=True)
    output = os.path.join(out_folder, name)

    if is3d:
        min_shape = [16, 128, 128]
        halo = [4, 32, 32]
    else:
        min_shape = [256, 256]
        halo = [32, 32]

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
        # training_data={"id": get_bioimageio_dataset_id("mitoem")},
        maintainers=[{"github_user": "constantinpape"}],
        min_shape=min_shape,
        halo=halo,
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

    checkpoints = glob("./checkpoints/s2d-lm-membrane-*")
    out_folder = f"./bio-models/v{args.version}"
    for ckpt in checkpoints:
        name = os.path.basename(ckpt)
        if(os.path.exists(os.path.join(out_folder, name))):
            print("Already exported::", name)
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
        export_enhancer(is3d, checkpoint=ckpt, version=args.version, name=name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--version", required=True, type=int)
    args = parser.parse_args()
    export_version(args)


if __name__ == "__main__":
    main()
