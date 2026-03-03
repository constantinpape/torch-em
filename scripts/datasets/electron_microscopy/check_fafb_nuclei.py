import os
import sys

import h5py
import numpy as np

sys.path.append("..")


def download_roi(center, size, mip=0):
    from cloudvolume import CloudVolume

    v1_image = "precomputed://https://storage.googleapis.com/flywire_em/aligned/v1"
    nuclei = "precomputed://https://storage.googleapis.com/neuroglancer/drosophila_v0/nucleus/v5_z_intp_intp/seg"

    c = np.asarray(center, np.int64)
    s = np.asarray(size,   np.int64)[::-1]
    start = c - (s // 2)
    end = start + s
    x0, y0, z0 = start.tolist()
    x1, y1, z1 = end.tolist()

    img_vol = CloudVolume(v1_image, mip=3, progress=True, bounded=False)
    seg_vol = CloudVolume(nuclei,   mip=0, progress=True, bounded=False)

    img = np.asarray(img_vol[x0:x1, y0:y1, z0:z1]).squeeze().T.copy()
    seg = np.asarray(seg_vol[x0:x1, y0:y1, z0:z1]).squeeze().T.copy()

    return img, seg


def prepare_cutouts():
    shape = (256, 512, 512)

    train_centers = [
        (11489, 9177, 3514),
        (10792, 7974, 3584),
        (10392, 7906, 2981),
        (6976, 7995, 3181),  # 4
        (10319, 8527, 2863),
        (10290, 6189, 2863),
        (11442, 4338, 2763),
        (9399, 8414, 3044)
    ]
    val_centers = [
        (22497, 6113, 3723),
        (22881, 5844, 2843),
    ]

    test_centers = [
        (21363, 9322, 2843),
        (24072, 11339, 2743),
        (22778, 9939, 2083),
        (22959, 8922, 2083),
    ]

    out_root = "./fafb_nuclei"

    def prepare_split(name, centers):
        out_folder = os.path.join(out_root, name)
        os.makedirs(out_folder, exist_ok=True)
        for i, center in enumerate(centers):
            out_path = os.path.join(out_folder, f"{name}_block_{i}.h5")
            if os.path.exists(out_path):
                continue
            img, seg = download_roi(center=center, size=shape)
            with h5py.File(out_path, mode="a") as f:
                f.create_dataset("raw", data=img, compression="gzip")
                f.create_dataset("labels/nuclei", data=seg, compression="gzip")

    prepare_split("train", train_centers)
    prepare_split("val", val_centers)
    prepare_split("test", test_centers)


def check_cutouts():
    import napari
    from glob import glob

    files = sorted(glob("./fafb_nuclei/**/*.h5", recursive=True))
    for ff in files:
        with h5py.File(ff) as f:
            raw = f["raw"][:]
            labels = f["labels/nuclei"][:]
        v = napari.Viewer()
        v.add_image(raw)
        v.add_labels(labels)
        v.title = ff
        napari.run()


def check_fafb_nuclei_loader():
    from torch_em.data.datasets.electron_microscopy import get_fafb_nuclei_loader
    from torch_em.util.debug import check_loader
    from util import ROOT

    loader = get_fafb_nuclei_loader(
        os.path.join(ROOT, "fafb-nuclei"), split="train", patch_shape=(8, 512, 512), batch_size=1, download=True
    )
    check_loader(loader, 8, instance_labels=True)


def main():
    # prepare_cutouts()
    # check_cutouts()
    check_fafb_nuclei_loader()


if __name__ == "__main__":
    main()
