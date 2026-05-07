import os
import sys

import h5py
import numpy as np

sys.path.append("..")


def _get_bb(center_nm, target_size, vol, mip):
    scale = vol.info["scales"][mip]["resolution"]
    # offset = vol.info["scales"][mip]["voxel_offset"]

    center = np.array([ce / sc for ce, sc in zip(center_nm, scale)])
    half = np.array(target_size) / 2
    return center - half, center + half


def download_roi(center, target_size, resolution, mip_em=3, mip_nuc=0):
    from cloudvolume import CloudVolume
    from cloudvolume.lib import Bbox

    em_src = "precomputed://https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/em"
    nuclei_src = "precomputed://https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/nuclei"

    em = CloudVolume(em_src, mip=mip_em, progress=False, bounded=False, fill_missing=True, autocrop=True)
    nuclei = CloudVolume(nuclei_src, mip=mip_nuc, progress=False, bounded=False, fill_missing=True, autocrop=True)

    center_nm = np.asarray([ce * res for ce, res in zip(center, resolution)], dtype=float)

    start_nuc, stop_nuc = _get_bb(center_nm, target_size, nuclei, mip_nuc)
    bbox_nuc = Bbox(start_nuc, stop_nuc)

    start_em, stop_em = _get_bb(center_nm, target_size, em, mip_em)
    bbox_em = Bbox(start_em, stop_em)

    em_vol = em.download(bbox_em, mip=mip_em).squeeze().T.copy()
    nuc_vol = nuclei.download(bbox_nuc, mip=mip_nuc).squeeze().T.copy()
    assert em_vol.shape == nuc_vol.shape

    # import napari
    # v = napari.Viewer()
    # v.add_image(em_vol)
    # v.add_labels(nuc_vol)
    # napari.run()

    return em_vol, nuc_vol


def prepare_cutouts():
    target_shape = (512, 512, 512)
    # target_shape = (256, 256, 128)
    resolution = (4, 4, 40)

    train_centers = [
        (100294, 141481, 21103),
        (69938, 126361, 21161),
        (71190, 144708, 20995),
        (73953, 204627, 20179),  # 4
        (95408, 255523, 20179),
        (119798, 267654, 20179),
        (130763, 255739, 19799),
        (151182, 247364, 19799),
    ]
    val_centers = [
        (187358, 135427, 19939),
        (287080, 158647, 18353),
    ]
    test_centers = [
        (209763, 107799, 19939),
        (250687, 103655, 19989),
        (279508, 144855, 19989),
        (292174, 117377, 19863),
    ]

    out_root = "./microns_nuclei"

    def prepare_split(name, centers):
        out_folder = os.path.join(out_root, name)
        os.makedirs(out_folder, exist_ok=True)
        for i, center in enumerate(centers):
            out_path = os.path.join(out_folder, f"{name}_block_{i}.h5")
            if os.path.exists(out_path):
                continue
            img, seg = download_roi(center=center, target_size=target_shape, resolution=resolution)
            print("Save", out_path)
            with h5py.File(out_path, mode="a") as f:
                f.create_dataset("raw", data=img, compression="gzip")
                f.create_dataset("labels/nuclei", data=seg, compression="gzip")

    prepare_split("train", train_centers)
    prepare_split("val", val_centers)
    prepare_split("test", test_centers)


def check_cutouts():
    import napari
    from glob import glob

    files = sorted(glob("./microns_nuclei/**/*.h5", recursive=True))
    for ff in files:
        with h5py.File(ff) as f:
            raw = f["raw"][:]
            labels = f["labels/nuclei"][:]
        v = napari.Viewer()
        v.add_image(raw)
        v.add_labels(labels)
        v.title = ff
        napari.run()


def check_microns_nuclei_loader():
    from torch_em.data.datasets.electron_microscopy import get_microns_nuclei_loader
    from torch_em.util.debug import check_loader
    from util import ROOT

    loader = get_microns_nuclei_loader(
        os.path.join(ROOT, "microns-nuclei"), split="train", patch_shape=(8, 512, 512), batch_size=1, download=True
    )
    check_loader(loader, 8, instance_labels=True)


def main():
    # prepare_cutouts()
    # check_cutouts()
    check_microns_nuclei_loader()


if __name__ == "__main__":
    main()
