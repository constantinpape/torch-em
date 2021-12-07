# create dataset rdfs to create bioimage.io datasets
import os

import imageio
import numpy as np
from ruamel.yaml import YAML
from skimage.transform import resize

yaml = YAML(typ="safe")


#
# Utils
#


def _resize(im, max_size):
    if im.ndim == 2:
        size = float(im.size)
        factor = max_size / size
        new_shape = tuple(int(sh * factor) for sh in im.shape)
        im = resize(im, new_shape)
    else:
        assert im.ndim == 3
        size = float(np.prod(im.shape[:-1]))
        factor = np.sqrt(max_size / size)
        new_shape = tuple(int(sh * factor) for sh in im.shape[:-1]) + im.shape[-1:]
        im = resize(im, new_shape)
    return im


def _package_rdf(rdf, name, doc):
    max_size = 256 * 256

    out_folder = os.path.join("./rdfs", name)
    if os.path.exists(os.path.join(out_folder, "rdf.yaml")):
        print("Dataset rdf for", name, "already exists")
        return

    print("Create dataset rdf for", name)
    os.makedirs(out_folder, exist_ok=True)

    covers = rdf["covers"]
    new_covers = []
    for ii, cover in enumerate(covers):
        im = imageio.imread(cover)
        size = im.size if im.ndim == 2 else np.prod(im.shape[:-1])
        if size > max_size:
            im = _resize(im, max_size)

        try:
            out = f"{out_folder}/{name}-cover{ii}.jpg"
            imageio.imwrite(out, im)
            new_covers.append(f"https://raw.githubusercontent.com/ilastik/bioimage-io-models/main/dataset_src/{name}-cover{ii}.jpg")
        except OSError:  # this is raised if trying to save an ARGB jpg
            os.remove(f"{out_folder}/{name}-cover{ii}.jpg")
            out = f"{out_folder}/{name}-cover{ii}.png"
            imageio.imwrite(out, im)
            new_covers.append(f"https://raw.githubusercontent.com/ilastik/bioimage-io-models/main/dataset_src/{name}-cover{ii}.png")

    with open(f"{out_folder}/{name}.md", "w") as f:
        f.write(doc)
    rdf["covers"] = new_covers
    with open(f"{out_folder}/rdf.yaml", "w") as f:
        yaml.dump(rdf, f)
    print("Successfully created dataset rdf for", name)


#
# rdf creation, tag guidelines:
# https://github.com/bioimage-io/bioimage.io/blob/main/site.config.json#L109-L155
#


def create_covid_if_rdf():
    doc = """#Covid-IF Training Data

Training data for cell and nucleus segmentation as well as infected cell classification of Covid19 in immunofluorescence.
The images are taken with a high-throughput microscope and were used as training data to set up an IF based CoV19 antibody assay [here](https://doi.org/10.1002/bies.202000257)."""
    rdf = {
        "name": "Covid-IF Training Data",
        "description": "Training data for cell and nucleus segmentation as well as infection classification in IF data of Covid-19 infected cells.",
        "cite": [
            {"doi": "https://doi.org/10.1002/bies.202000257",
             "text": "Pape, Remme et al."}
        ],
        "authors": [
            {"name": "Constantin Pape"}
        ],
        "documentation": "https://raw.githubusercontent.com/ilastik/bioimage-io-models/main/dataset_src/covid-if.md",
        "tags": ["high-content-imaging", "fluorescence-light-microscopy", "2D", "cells", "nuclei", "covid19",
                 "semantic-segmentation", "instance-segmentation"],
        "source": "https://zenodo.org/record/5092850",
        "covers": [
            "https://sandbox.zenodo.org/record/881843/files/cover0.jpg",
            "https://sandbox.zenodo.org/record/881843/files/cover1.jpg",
            "https://sandbox.zenodo.org/record/881843/files/cover2.jpg",
        ],
        "type": "dataset",
        "license": "CC-BY-4.0"
    }
    _package_rdf(rdf, "covid-if", doc)


def create_cremi_rdf():
    doc = """#CREMI Neuron Segmentation Challenge Training Data

Training data for instance segmentation of neurons in 3d EM data. From the [CREMI Miccai challenge](https://cremi.org/).
    """
    rdf = {
        "name": "CREMI: MICCAI Challenge on Circuit Reconstruction from Electron Microscopy Images",
        "description": "Training data from the challenge on 3d EM segmentation on neuronal processes.",
        "cite": [
            {"text": "Jan Funke, Stephan Saalfeld, Davi Bock, Srini Turaga, Eric Perlman",
             "url": "https://cremi.org/"}
        ],
        "authors": [
            {"name": "Constantin Pape"}
        ],
        "documentation": "https://raw.githubusercontent.com/ilastik/bioimage-io-models/main/dataset_src/cremi.md",
        "tags": ["electron-microscopy", "brain", "neurons", "instance-segmentation", "cremi-challenge", "3D"],
        "source": "https://cremi.org/",
        "covers": [
            "https://sandbox.zenodo.org/record/881917/files/cover0.png",
            "https://sandbox.zenodo.org/record/881917/files/cover1.png",
            "https://sandbox.zenodo.org/record/881917/files/cover2.png",
        ],
        "type": "dataset",
    }
    _package_rdf(rdf, "cremi", doc)


def create_dsb_rdf():
    doc = """# Nucleus Segmentation Training Data

Traning data for instance segmentation of nuclei. This data is a subset of the data from the [2018 Kaggle Data Science Bowl dataset
for nucleus segmentation](https://bioimage.io/#/?type=application&id=notebook_stardist_2d_zerocostdl4mic).
This subset has been used for training the [StarDist](https://github.com/stardist/stardist) nucleus segmentation model and other nucleus segmentation models on bioimagei.io."""
    rdf = {
        "name": "DSB Nucleus Segmentation Training Data",
        "description": "Subset of the nucleus segmentation training data from the 2018 Kaggle Data Science Bowl.",
        "cite": [
            {"doi": "https://doi.org/10.1038/s41592-019-0612-7",
             "text": "Caicedo et al."}
        ],
        "authors": [
            {"name": "Constantin Pape"}
        ],
        "documentation": "https://raw.githubusercontent.com/ilastik/bioimage-io-models/main/dataset_src/dsb.md",
        "tags": ["nuclei", "instance-segmentation", "fluorescence-light-microscopy", "dsb-challenge", "2D"],
        "source": "https://github.com/stardist/stardist/releases/download/0.1.0/dsb2018.zip",
        "covers": [
            "https://storage.googleapis.com/kaggle-media/competitions/dsb-2018/dsb.jpg"
        ],
        "type": "dataset",
        "license": "CC0-1.0"
    }
    _package_rdf(rdf, "dsb", doc)


def create_isbi2012_rdf():
    doc = """#ISBI2012 Neuron Segmentation Challenge Training Data

Traning data for instance segmentation of neurons in EM. From the [2012 ISBI EM Segmentation Challenge](https://brainiac2.mit.edu/isbi_challenge/)."""
    rdf = {
        "name": "ISBI Challenge: Segmentation of neuronal structures in EM stacks",
        "description": "Training data from challenge on 2d EM segmentation of neuronal processes.",
        "cite": [
            {"text": "Arganda-Carreras et al.",
             "doi": "https://doi.org/10.3389/fnana.2015.00142"}
        ],
        "authors": [
            {"name": "Constantin Pape"}
        ],
        "documentation": "https://raw.githubusercontent.com/ilastik/bioimage-io-models/main/dataset_src/isbi.md",
        "tags": ["electron-microscopy", "brain", "neurons", "instance-segmentation", "2D", "isbi2012-challenge"],
        "source": "https://oc.embl.de/index.php/s/sXJzYVK0xEgowOz/download",
        "covers": [
            "https://sandbox.zenodo.org/record/883893/files/cover0.jpg",
            "https://sandbox.zenodo.org/record/883893/files/cover1.gif"
        ],
        "type": "dataset",
    }
    _package_rdf(rdf, "isbi2012", doc)


def create_livecell_rdf():
    doc = """#Livecell Training Data

Traning data for cell segmentation in live-cell / phase-contrast imaging. From [LIVECell—A large-scale dataset for label-free live cell segmentation](https://doi.org/10.1038/s41592-021-01249-6)."""
    rdf = {
        "name": "LIVECell",
        "description": "LIVECell—A large-scale dataset for label-free live cell segmentation",
        "cite": [
            {"text": "Edlund et al.",
             "doi": "https://doi.org/10.1038/s41592-021-01249-6"}
        ],
        "authors": [
            {"name": "Constantin Pape"}
        ],
        "documentation": "https://raw.githubusercontent.com/ilastik/bioimage-io-models/main/dataset_src/livecell.md",
        "tags": ["2D", "transmission-light-microscopy", "label-free", "cells", "instance-segmentation"],
        "source": "https://sartorius-research.github.io/LIVECell/",
        "covers": [
            "https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41592-021-01249-6/MediaObjects/41592_2021_1249_Fig1_HTML.png",
        ],
        "type": "dataset",
        "license": "CC-BY-NC-4.0"
    }
    _package_rdf(rdf, "livecell", doc)


# DONE
def create_mitoem_rdf():
    doc = """#MitoEM Challenge Training Data

This is the training data for the [MitoEM Large-scale 3D Mitochondria Instance Segmentation Challenge](https://mitoem.grand-challenge.org/).
It contains two 30x30x30 micron datasets (1000x4096x4096 voxels at 30x8x8 nm resolution) of rat and human cortex with
mitochondria instance labels."""
    rdf = {
        "name": "MitoEM Challenge: Large-scale 3D Mitochondria Instance Segmentation",
        "description": "Training data for mitochondria segmentation in 3d EM.",
        "cite": [
            {"doi": "https://doi.org/10.1007/978-3-030-59722-1_7",
             "text": "Donglai Wei et al."}
        ],
        "authors": [
            {"name": "Constantin Pape"}
        ],
        "documentation": "https://raw.githubusercontent.com/ilastik/bioimage-io-models/main/dataset_src/mitoem.md",
        "tags": ["mitochondria", "electron-microscopy", "3D", "mito-em-challenge", "instance-segmentation"],
        "source": "https://mitoem.grand-challenge.org/",
        "covers": [
            "https://grand-challenge-public-prod.s3.amazonaws.com/b/566/banner.x10.jpeg",
            "https://grand-challenge-public-prod.s3.amazonaws.com/i/2020/10/27/mitoEM_teaser.png"
        ],
        "type": "dataset",
    }
    _package_rdf(rdf, "mitoem", doc)


def create_platy_rdf():
    doc = """#Platynereis EM Segmentation Training Data

Training data for segmentation of cellular membranes, nuclei, cuticle and cilia in Platynereis dumerilii.
The training data is extracted from a whole body EM volume of a Platynereis larva that was used
to train networks for segmentation in [Whole-body integration of gene expression and single-cell morphology](https://doi.org/10.1016/j.cell.2021.07.017).
For each organelle several blocks of training data are available, extracted at dfferent resolutions:
20x20x25nm for cellular membranes, 80x80x100nm for nuclei, 40x40x50nm for cuticle and 10x10x25nm for cilia."""
    rdf = {
        "name": "Platynereis EM Traning Data",
        "description": "Training data for EM segmentation of cellular membranes, nuclei, cuticle and cilia in Platynereis.",
        "cite": [
            {"doi": "https://doi.org/10.1016/j.cell.2021.07.017",
             "text": "Vergara, Pape, Meechan et al."}
        ],
        "authors": [
            {"name": "Constantin Pape"}
        ],
        "documentation": "https://raw.githubusercontent.com/ilastik/bioimage-io-models/main/dataset_src/platy.md",
        "tags": ["electron-microscopy", "platynereis", "cells", "cilia", "nuclei", "instance-segmentation", "3D"],
        "source": "https://doi.org/10.5281/zenodo.3675220",
        "covers": [
            "https://sandbox.zenodo.org/record/881899/files/cover0.png"
        ],
        "type": "dataset",
        "license": "CC-BY-4.0"
    }
    _package_rdf(rdf, "platy", doc)


# Skipped TODO discuss with Adrian how to re-upload plantseg models
def create_ovules_rdf():
    rdf = {
        "name": "Arabidopsis thaliana ovules - confocal",
        "description": "Ovules - confocal volumetric stacks with voxel size: (0.235x0.075x0.075 µm^3) (ZYX). Courtesy of Kay Schneitz lab, School of Life Sciences, Technical University of Munich, Germany.",
        "cite": [
            {"doi": "https://doi.org/10.7554/eLife.57613",
             "text": "Adrian Wolny et al."}
        ],
        "authors": [
            {"name": "Constantin Pape"}
        ],
        "documentation": "https://osf.io/uzq3w/wiki/home/",
        "tags": ["confocal", "lm-segmentation", "arabidopsis", "arabidopsis thaliana", "ovules", "plants"],
        "source": "https://osf.io/w38uf/",
        "covers": [
            "https://raw.githubusercontent.com/hci-unihd/plant-seg/master/Documentation-GUI/images/main_figure_nologo.png"
        ],
        "type": "dataset",
        "license": "CC-BY-4.0"
    }
    _package_rdf(rdf, "ovules")


if __name__ == '__main__':
    os.makedirs("./rdfs", exist_ok=True)

    create_covid_if_rdf()
    create_cremi_rdf()
    create_dsb_rdf()
    create_isbi2012_rdf()
    create_livecell_rdf()
    create_mitoem_rdf()
    create_platy_rdf()

    # Not sure what to do about ovules
    # create_ovules_rdf()
