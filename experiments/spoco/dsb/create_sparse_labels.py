import os
from glob import glob

import imageio
import numpy as np
import vigra
from tqdm import tqdm


def create_sparse_labels(in_folder, out_folder, label_fraction):
    os.makedirs(out_folder, exist_ok=True)
    mask_paths = glob(os.path.join(in_folder, "*.tif"))

    for path in tqdm(mask_paths, desc=f"Create label fraction {label_fraction}"):
        mask = imageio.imread(path)
        ids = np.unique(mask)
        assert ids[0] == 0
        ids = ids[1:]
        n_ids = len(ids)
        keep_ids = np.random.choice(ids, size=int(label_fraction * n_ids), replace=False)
        mask[~np.isin(mask, keep_ids)] = 0
        mask = mask.astype("uint32")
        vigra.analysis.relabelConsecutive(mask, out=mask, start_label=1, keep_zeros=True)

        fname = os.path.basename(path)
        out_path = os.path.join(out_folder, fname)
        imageio.imwrite(out_path, mask)


def create_all_sparse_labels(label_root):
    label_folder = os.path.join(label_root, "masks")
    np.random.seed(42)
    label_fractions = [0.1, 0.25, 0.4, 0.5, 0.75]
    for frac in label_fractions:
        out_folder = os.path.join(label_root, "sparse_labels", f"fraction-{frac}")
        create_sparse_labels(label_folder, out_folder, frac)


def main():
    data_root = "/home/pape/Work/data/dsb"
    create_all_sparse_labels(os.path.join(data_root, "train"))
    create_all_sparse_labels(os.path.join(data_root, "test"))


if __name__ == "__main__":
    main()
