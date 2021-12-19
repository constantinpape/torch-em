import os
from glob import glob
import torch_em
from torch_em.shallow2deep import prepare_shallow2deep, BoundaryTransform


def prepare_shallow2deep_isbi(args, out_folder):
    patch_shape_min = [1, 256, 256]
    patch_shape_max = [1, 512, 512]

    raw_transform = torch_em.transform.raw.normalize
    label_transform = BoundaryTransform(ndim=2)

    prepare_shallow2deep(
        raw_paths=args.input, raw_key="volumes/raw",
        label_paths=args.input, label_key="volumes/labels/neuron_ids_3d",
        patch_shape_min=patch_shape_min, patch_shape_max=patch_shape_max,
        n_forests=args.n_rfs, n_threads=args.n_threads,
        output_folder=out_folder, ndim=2,
        raw_transform=raw_transform, label_transform=label_transform,
    )


def train_shallow2deep(args):
    # TODO find a version scheme for names depending on args and existing versions
    name = "isbi2d"

    # check if we need to train the rfs for preparation
    rf_folder = os.path.join("checkpoints", name, "rfs")
    have_rfs = len(glob(os.path.join(rf_folder, "*.pkl"))) == args.n_rfs
    if not have_rfs:
        prepare_shallow2deep_isbi(args, rf_folder)


if __name__ == "__main__":
    parser = torch_em.util.parser_helper()
    parser.add_argument("--n_rfs", type=int, default=8)  # TODO more rfs
    parser.add_argument("--n_threads", type=int, default=8)
    args = parser.parse_args()
    train_shallow2deep(args)
