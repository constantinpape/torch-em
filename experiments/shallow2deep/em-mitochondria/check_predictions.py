import argparse
from elf.io import open_file
from torch_em.shallow2deep.shallow2deep_eval import load_predictions


def check_predictions(version):
    if version == 1:
        data_path = "/g/kreshuk/pape/Work/data/mito_em/data/crops/crop_test.h5"
        label_key = "labels"
    elif version == 2:
        data_path = "/g/kreshuk/data/VNC/data_labeled_mito.h5"
        label_key = "label"

    print("Loading data ...")
    with open_file(data_path, "r") as f:
        dsr = f["raw"]
        dsr.n_threads = 8
        raw = dsr[:]

        dsl = f[label_key]
        dsl.n_threads = 8
        labels = dsl[:]

    prediction_path = f"./bio-models/v{version}/prediction.h5"
    print("Loading predictions ...")
    predictions = load_predictions(prediction_path, n_threads=8)

    print("Starting viewer ...")
    import napari
    v = napari.Viewer()
    v.add_image(raw)
    v.add_labels(labels)
    for name, pred in predictions.items():
        v.add_image(pred, name=name)
    napari.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--version", required=True, type=int)
    args = parser.parse_args()
    check_predictions(args.version)
