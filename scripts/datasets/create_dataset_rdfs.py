# create dataset rdfs to create bioimage.io datasets

import os
from shutil import rmtree, make_archive

import imageio
import numpy as np
from ruamel.yaml import YAML
from skimage.transform import resize

yaml = YAML(typ="safe")


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


def _package_rdf(rdf, name):
    max_size = 256 * 256

    if os.path.exists("tmp"):
        rmtree("tmp")
    os.makedirs("tmp")

    covers = rdf["covers"]
    new_covers = []
    for ii, cover in enumerate(covers):
        im = imageio.imread(cover)
        size = im.size if im.ndim == 2 else np.prod(im.shape[:-1])
        if size > max_size:
            im = _resize(im, max_size)

        try:
            out = f"./tmp/cover{ii}.jpg"
            imageio.imwrite(out, im)
            new_covers.append(f"cover{ii}.jpg")
        except OSError:  # this is raised if trying to save an ARGB jpg
            os.remove(f"./tmp/cover{ii}.jpg")
            out = f"./tmp/cover{ii}.png"
            imageio.imwrite(out, im)
            new_covers.append(f"cover{ii}.png")

    rdf["covers"] = new_covers
    with open("./tmp/rdf.yaml", "w") as f:
        yaml.dump(rdf, f)

    make_archive(f"./rdfs/{name}", "zip", "tmp")
    if os.path.exists("tmp"):
        rmtree("tmp")


def create_isbi2012_rdf():
    rdf = {
        "name": "ISBI Challenge: Segmentation of neuronal structures in EM stacks",
        "description": "First challenge on 2d segmentation of neuronal processes in EM images. Organised as part of the ISBI2012 conference.",
        "cite": [
            {"text": "Arganda-Carreras et al.",
             "doi": "https://doi.org/10.3389/fnana.2015.00142"}
        ],
        "authors": [
            {"name": "Constantin Pape"}
        ],
        "documentation": "http://brainiac2.mit.edu/isbi_challenge/home",
        "tags": ["neuron-segmentation", "em-segmentation", "isbi2012-challenge"],
        "source": "http://brainiac2.mit.edu/isbi_challenge/home",
        "covers": [
            "http://brainiac2.mit.edu/isbi_challenge/sites/default/files/Challenge-ISBI-2012-sample-image.png",
            # not sure if we support gifs
            # "http://brainiac2.mit.edu/isbi_challenge/sites/default/files/Challenge-ISBI-2012-Animation-Input-Labels.gif"
        ],
        "type": "dataset",
        "license": "CC-BY-4.0"
    }
    _package_rdf(rdf, "isbi2012")


def create_cremi_rdf():
    rdf = {
        "name": "CREMI: MICCAI Challenge on Circuit Reconstruction from Electron Microscopy Images",
        "description": "The goal of this challenge is to evaluate algorithms for automatic reconstruction of neurons and neuronal connectivity from serial section electron microscopy data.",
        "cite": [
            {"text": "Jan Funke, Stephan Saalfeld, Davi Bock, Srini Turaga, Eric Perlman",
             "url": "https://cremi.org/"}
        ],
        "authors": [
            {"name": "Constantin Pape"}
        ],
        "documentation": "https://cremi.org/",
        "tags": ["neuron-segmentation", "em-segmentation", "cremi-challenge"],
        "source": "https://cremi.org/",
        "covers": [
            "https://cremi.org/static/img/sample_A_preview.png",
            "https://cremi.org/static/img/sample_B_preview.png",
            "https://cremi.org/static/img/sample_C_preview.png"
        ],
        "type": "dataset",
        "license": "CC-BY-4.0"
    }
    _package_rdf(rdf, "cremi")


def create_mitoem_rdf():
    rdf = {
        "name": "MitoEM Challenge: Large-scale 3D Mitochondria Instance Segmentation",
        "description": "The task is the 3D mitochondria instance segmentation on two 30x30x30 um datasets, 1000x4096x4096 in voxels at 30x8x8 nm resolution.",
        "cite": [
            {"doi": "https://doi.org/10.1007/978-3-030-59722-1_7",
             "text": "Donglai Wei et al."}
        ],
        "authors": [
            {"name": "Constantin Pape"}
        ],
        "documentation": "https://mitoem.grand-challenge.org/",
        "tags": ["mitochondria-segmentation", "em-segmentation", "mito-em-challenge"],
        "source": "https://mitoem.grand-challenge.org/",
        "covers": [
            "https://grand-challenge-public-prod.s3.amazonaws.com/b/566/banner.x10.jpeg",
            "https://grand-challenge-public-prod.s3.amazonaws.com/i/2020/10/27/mitoEM_teaser.png"
        ],
        "type": "dataset",
        "license": "CC-BY-4.0"
    }
    _package_rdf(rdf, "mitoem")


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


def create_platy_rdf():
    rdf = {
        "name": "Platynereis EM traning data",
        "description": "Training data for EM cell, nuclei and organelle segmentation in Platynereis dumerilii. Contains training data for cellular membranes, nuclei, cuticle and cilia.",
        "cite": [
            {"url": "https://www.biorxiv.org/content/10.1101/2020.02.26.961037v1.abstract",
             "text": "Vergara, Pape, Meechan et al."}
        ],
        "authors": [
            {"name": "Constantin Pape"}
        ],
        "documentation": "https://www.biorxiv.org/content/10.1101/2020.02.26.961037v1.abstract",
        "tags": ["em-segmentation", "platynereis", "platynereis dumerilii", "cell-segmentation", "cilia-segmentation", "nucleus-segmentation"],
        "source": "https://osf.io/w38uf/",
        "covers": [
            "./fig2_compact_300.png"
        ],
        "type": "dataset",
        "license": "CC-BY-4.0"
    }
    _package_rdf(rdf, "platy")


def create_covid_if_rdf():
    rdf = {
        "name": "CovidIf training data",
        "description": "Training data for cell, nucleus and infection classification in IF data of Covid-19 infected cells.",
        "cite": [
            {"doi": "https://doi.org/10.1002/bies.202000257",
             "text": "Pape, Remme et al."}
        ],
        "authors": [
            {"name": "Constantin Pape"}
        ],
        "documentation": "https://onlinelibrary.wiley.com/doi/full/10.1002/bies.202000257",
        "tags": ["cell-segmentation", "nucleus-segmentation", "high-througput-microscopy", "covid-19", "sars-cov-2"],
        "source": "https://zenodo.org/record/5092850#.YPBInTqxVH4",
        "covers": [
            "bies202000257-fig-0001-m.jpg",
            "bies202000257-fig-0002-m.jpg",
            "bies202000257-fig-0003-m.jpg"
        ],
        "type": "dataset",
        "license": "CC-BY-4.0"
    }
    _package_rdf(rdf, "covid-if")


def create_dsb_rdf():
    rdf = {
        "name": "DSB Nucleus Segmentation Training Data",
        "description": "Subset of the nucleus segmentation training data provided by the 2018 Kaggle Data Science Bowl.",
        "cite": [
            {"url": "https://www.nature.com/articles/s41592-019-0612-7#Ack1",
             "text": "Caicedo et al."}
        ],
        "authors": [
            {"name": "Constantin Pape"}
        ],
        "documentation": "https://www.kaggle.com/c/data-science-bowl-2018",
        "tags": ["nucleus-segmentation", "dsb", "dsb2018"],
        "source": "",
        "covers": [
            "https://storage.googleapis.com/kaggle-media/competitions/dsb-2018/dsb.jpg"
        ],
        "type": "dataset",
        "license": "CC-BY-4.0"
    }
    _package_rdf(rdf, "dsb")


if __name__ == '__main__':
    os.makedirs("./rdfs", exist_ok=True)
    # create_isbi2012_rdf()
    # create_cremi_rdf()
    # create_mitoem_rdf()
    # create_covid_if_rdf()
    # create_platy_rdf()
    # create_ovules_rdf()
    create_dsb_rdf()
