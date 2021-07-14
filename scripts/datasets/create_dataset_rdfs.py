# create dataset rdfs to create bioimage.io datasets
import os
from ruamel.yaml import YAML
yaml = YAML(typ="safe")


def create_isbi2012_rdf():
    rdf = {
        "id": "isbi2012_em_segmentation_challenge_dataset",
        "name": "ISBI Challenge: Segmentation of neuronal structures in EM stacks",
        "description": "First challenge on 2d segmentation of neuronal processes in EM images. Organised as part of the ISBI2012 conference.",
        "cite": {
            "doi": "https://doi.org/10.3389/fnana.2015.00142",
            "authors": ["Ignacio Arganda-Carreras et al."]
        },
        "authors": ["Constantin Pape"],
        "documentation": "http://brainiac2.mit.edu/isbi_challenge/home",
        "tags": ["neuron-segmentation", "em-segmentation", "isbi2012-challenge"],
        "source": "http://brainiac2.mit.edu/isbi_challenge/home",
        "covers": [
            "http://brainiac2.mit.edu/isbi_challenge/sites/default/files/Challenge-ISBI-2012-sample-image.png",
            "http://brainiac2.mit.edu/isbi_challenge/sites/default/files/Challenge-ISBI-2012-Animation-Input-Labels.gif"
        ]
    }
    with open("./rdfs/isbi2012.yaml", "w") as f:
        yaml.dump(rdf, f)


def create_cremi_rdf():
    rdf = {
        "id": "cremi_em_segmentation_challenge_dataset",
        "name": "CREMI: MICCAI Challenge on Circuit Reconstruction from Electron Microscopy Images",
        "description": "The goal of this challenge is to evaluate algorithms for automatic reconstruction of neurons and neuronal connectivity from serial section electron microscopy data.",
        "cite": {
            "url": "https://cremi.org/",
            "authors": ["Jan Funke", "Stephan Saalfeld", "Davi Bock", "Srini Turaga", "Eric Perlman"]
        },
        "authors": ["Constantin Pape"],
        "documentation": "https://cremi.org/",
        "tags": ["neuron-segmentation", "em-segmentation", "cremi-challenge"],
        "source": "https://cremi.org/",
        "covers": [
            "https://cremi.org/static/img/sample_A_preview.png",
            "https://cremi.org/static/img/sample_B_preview.png",
            "https://cremi.org/static/img/sample_C_preview.png"
        ]
    }
    with open("./rdfs/cremi.yaml", "w") as f:
        yaml.dump(rdf, f)


def create_mitoem_rdf():
    rdf = {
        "id": "mito_em_segmentation_challenge_dataset",
        "name": "MitoEM Challenge: Large-scale 3D Mitochondria Instance Segmentation",
        "description": "The task is the 3D mitochondria instance segmentation on two 30x30x30 um datasets, 1000x4096x4096 in voxels at 30x8x8 nm resolution.",
        "cite": {
            "doi": "https://doi.org/10.1007/978-3-030-59722-1_7",
            "authors": ["Donglai Wei et al."]
        },
        "authors": ["Constantin Pape"],
        "documentation": "https://mitoem.grand-challenge.org/",
        "tags": ["mitochondria-segmentation", "em-segmentation", "mito-em-challenge"],
        "source": "https://mitoem.grand-challenge.org/",
        "covers": [
            "https://grand-challenge-public-prod.s3.amazonaws.com/b/566/banner.x10.jpeg",
            "https://grand-challenge-public-prod.s3.amazonaws.com/i/2020/10/27/mitoEM_teaser.png"
        ]
    }
    with open("./rdfs/mito_em.yaml", "w") as f:
        yaml.dump(rdf, f)


def create_ovules_rdf():
    pass


def create_platy_rdf():
    pass


def create_covid_if_rdf():
    pass


if __name__ == '__main__':
    os.makedirs("./rdfs", exist_ok=True)
    create_isbi2012_rdf()
    create_cremi_rdf()
    create_mitoem_rdf()
