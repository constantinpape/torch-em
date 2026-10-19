import os
import argparse

import matplotlib
matplotlib.use("Agg")

from torch_em.data.datasets.electron_microscopy.parlakgul_liver_eval import get_parlakgul_liver_eval_loader  # noqa
from torch_em.util.debug import check_loader  # noqa

CIDAS_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_parlakgul_liver_eval(path, samples, label_choice, semantic, return_prediction, n_samples, save_path):
    loader = get_parlakgul_liver_eval_loader(
        path, patch_shape=(25, 150, 150), batch_size=1, samples=samples, label_choice=label_choice,
        semantic=semantic, return_prediction=return_prediction, download=True,
    )
    check_loader(loader, n_samples, instance_labels=not semantic, plt=save_path is not None, save_path=save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", default=os.path.join(CIDAS_ROOT, "parlakgul_liver_eval"), help="Where to save the data."
    )
    parser.add_argument("--samples", nargs="+", default=["6461", "6464", "9430", "1857"], help="The liver samples.")
    parser.add_argument("--label_choice", default="mito", help="The organelle: mito or lipid_droplet.")
    parser.add_argument("--semantic", action="store_true", help="Binary foreground instead of instance ids.")
    parser.add_argument("--prediction", action="store_true", help="Show the model prediction instead of the GT.")
    parser.add_argument("-n", "--n_samples", type=int, default=8, help="The number of samples to display.")
    parser.add_argument("--save_path", default=None, help="Save a matplotlib figure here instead of using napari.")
    args = parser.parse_args()
    check_parlakgul_liver_eval(
        args.path, args.samples, args.label_choice, args.semantic, args.prediction, args.n_samples, args.save_path
    )


if __name__ == "__main__":
    main()
