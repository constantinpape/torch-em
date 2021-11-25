import argparse
import os
from subprocess import run


def _export(folder, ckpt, test_input, export_affinities, force_export, name=None):
    n_levels = folder.count("/") + 1
    if name is None:
        name = os.path.split(folder)[1] if n_levels > 1 else folder

    if export_affinities:
        out_folder = f"exported_models_mws/{name}"
    else:
        out_folder = f"exported_models/{name}"

    if os.path.exists(out_folder) and not force_export:
        print(name, "is already exported")
        return
    print("Export",  name, "...")

    rel = "/".join(n_levels * [".."])
    out_folder = f"{rel}/{out_folder}"

    cmd = ["python", "export_bioimageio_model.py",
           "-c", ckpt,
           "-i", test_input,
           "-o", out_folder,
           "-a", "0" if export_affinities else "1"]
    if export_affinities:
        cmd.extend(["-f", "torchscript", "onnx"])
    else:
        cmd.extend(["-f", "torchscript"])
    cwd = os.getcwd()
    os.chdir(folder)
    run(cmd)
    os.chdir(cwd)


def export_boundary_models(export_affinities, force_export, include_models):
    if export_affinities:
        os.makedirs("exported_models_mws", exist_ok=True)
    else:
        os.makedirs("exported_models", exist_ok=True)

    models = [
        ("covid-if", "checkpoints/covid-if-affinity-model",
         "/scratch/pape/covid-if/gt_image_000.h5"),
        ("neuron-segmentation/cremi", "checkpoints/affinity_model_default",
         "/scratch/pape/cremi/sample_C_20160501.hdf"),
        ("dsb", "checkpoints/dsb-affinity-model",
         "/scratch/pape/dsb/test/images/0bda515e370294ed94efd36bd53782288acacb040c171df2ed97fd691fc9d8fe.tif"),
        ("neuron-segmentation/isbi2012", "checkpoints/affinity-model",
         "/g/kreshuk/data/isbi2012_challenge/isbi2012_test_volume.h5"),
        ("livecell", "checkpoints/livecell-affinity-model",
         "/scratch/pape/livecell/images/livecell_train_val_images/BV2/BV2_Phase_B4_1_00d04h00m_2.tif"),
        ("mito-em", "checkpoints/affinity_model_default_human_rat",
         "/scratch/pape/mito_em/data/human_test.n5"),
        ("monuseg", "checkpoints/monuseg-affinity-model",
         "/g/kreshuk/pape/Work/data/monuseg/images/TCGA-18-5592-01Z-00-DX1.tif"),
        ("plantseg", "ovules/checkpoints/affinity_model2d",
         "/g/kreshuk/wolny/Datasets/Ovules/GT2x/val/N_420_ds2x.h5", "ovules"),
        ("platynereis", "cells/checkpoints/affinity_model",
         "/scratch/pape/platy/membrane/train_data_membrane_09.n5", "platy-cell"),
        ("platynereis", "nuclei/checkpoints/affinity_model",
         "/scratch/pape/platy/nuclei/train_data_nuclei_12.h5", "platy-nucleus")
    ]

    for model in models:
        if include_models is not None and model[0] not in include_models:
            continue
        name = model[3] if len(model) == 4 else None
        _export(model[0], model[1], model[2], export_affinities, force_export, name=name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--affinities", type=int, default=0)
    parser.add_argument("-f", "--force", type=int, default=0)
    parser.add_argument("-i", "--include", type=str, nargs="+", default=None)
    args = parser.parse_args()
    export_boundary_models(bool(args.affinities), bool(args.force), args.include)


if __name__ == "__main__":
    main()
