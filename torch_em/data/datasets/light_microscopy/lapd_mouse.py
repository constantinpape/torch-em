"""The LAPD Mouse dataset (Lung Anatomy + Particle Deposition mouse archive) contains 3D cryomicrotome
fluorescence imaging volumes of 34 mouse lungs with annotations of the airway tree.

The raw data is the autofluorescence channel of the imaging cryomicrotome, which shows the anatomical structures
(lung, fissures, airway walls). The full resolution volumes have a voxel size of about 9 x 9 x 9.5 um and a size of
about 2000 x 1500 x 2500 voxels (more than 10 GB per mouse). The archive also provides versions downsampled by a
factor of 2 ('sub2', about 1.5 GB per mouse) and 4 ('sub4', about 200 MB per mouse), which can be chosen via the
`resolution` argument.

The labels are the airway segment labelmaps: every airway segment (branch) from the trachea to the terminal
bronchi has its own id (1 to about 1800 per mouse), which corresponds to the segment ids in the airway tree tables
of the archive. Use `labels > 0` to obtain a binary airway mask. The labelmaps are provided at full resolution only
and are downsampled by strided subsampling to match the chosen raw resolution.

The raw data and labels are converted to a hdf5 file per mouse, with the keys 'raw' (uint16) and 'labels' (uint16).

The dataset is located at https://cebs-ext.niehs.nih.gov/cahs/report/lapd/web-download-links.

This dataset is from the archive https://doi.org/10.25820/9arg-9w56 (Beichel et al., University of Iowa, 2019).
Please cite it if you use this dataset in your research.
"""

import os
import zlib
from typing import Union, Tuple, Literal, List, Optional, Sequence

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://cebs-ext.niehs.nih.gov/cahs/file/download/lapd/{mouse_id}/{filename}"

MOUSE_IDS = [f"m{i:02d}" for i in range(1, 35)]

RESOLUTIONS = {"full": ("", 1), "sub2": ("Sub2", 2), "sub4": ("Sub4", 4)}

# Checksums are only available for the 'sub4' raw volumes and the labels; 'full' and 'sub2' are not verified.
CHECKSUMS = {
    "m01": {
        "sub4": "af22d8ce03860098926bad1130518a3dd612f8d6f471341bd67b381255d6da1c",
        "labels": "c12945ee05bd616a58290bdf7f3aee5fe498511ff42a77ebadabff441ebd6666",
    },
    "m02": {
        "sub4": "0563ee1e984d4d4a9fabb49bc8615d8ae302c6422f835b459a36335a73f338a1",
        "labels": "82b6d22b49e87bb53b4c82e69a6c0fedf32cb1567ba41857e1eeb945d414f95d",
    },
    "m03": {
        "sub4": "83663a32b770066edd2cfe5524e5b0675f41291c2cb4046a9abf4c426b588898",
        "labels": "3195006484650331a19895e41dbec1a0fe0134b4ed5d93a71f61c225a772fe1b",
    },
    "m04": {
        "sub4": "35ded7031395af0b1f65a2b75184173aff95e2e4022df9a7815b4e1fd1c30916",
        "labels": "995740d07ef38dded21401ceb4b1d1eed03a6cb0456d3c25471599509e687b71",
    },
    "m05": {
        "sub4": "72736c2caf3f2c88fe0740e6b50d462e383187612f67758d8d4c2d0ccda7b072",
        "labels": "b086d2e09f206354578ef19b5d9f4ccd978db2bf68924de07d84a74baf2de036",
    },
    "m06": {
        "sub4": "71827c4d3429892592b3d5b20d96ba236036a87cf358f080efb0e7dd1381056e",
        "labels": "8bd0f6b3c982653ed7bac790dc886374b3b8072f2a41a119c5a009447b1842f8",
    },
    "m07": {
        "sub4": "9ae21f57649f6402b5abf7b6262fdf3c2385d95970ff60b7f29f38c5c8e90dc9",
        "labels": "33d3d23f026d37cbf5c5f6394f31f3704166b5b56654b5bb4778e54e3bd45504",
    },
    "m08": {
        "sub4": "772d1ecfb5b1c618d5a42c43adef06adaae4f5735a2c443c7314fb9f0f82edb3",
        "labels": "00cbec1610f5cf69c3fe46e402eef619e2c3acefc01a5059cb9acd65c8427b70",
    },
    "m09": {
        "sub4": "375243048f41773a034091c10e1804293d25548fb34e8e227783f087f76e329f",
        "labels": "ff20215d2c31e615f907e471b15100142139c107eff7715f1b7d32268f37f213",
    },
    "m10": {
        "sub4": "f5d6d149110e4b6561a81f5eb81c274b626bed98cd4283dec6f2009002f10330",
        "labels": "13c6d12947a4a315d7cc4e68fe36ae60fcac2eb5c5e2af72337993eba16c10d0",
    },
    "m11": {
        "sub4": "d10a5a5dbc55b97902d5b041866e89d87b39efbb2f6b142b672e97c02734e10a",
        "labels": "9c3692147181daa16c50cf0314fec82b658922aca62eebcd10550462f6a8155d",
    },
    "m12": {
        "sub4": "7ff55b74f8c4c46ce09fec86580c44612105d21e8f124eb809c3a11c5f613d5f",
        "labels": "4e66ba6cc52d44740c7186ecafba6332d22d03707a14b933535acac4bea91d0f",
    },
    "m13": {
        "sub4": "21689e429107b1927774076a421033bcf372ac7c4c19604b6ae011f7ea9d5bbd",
        "labels": "96b826ba0026f24c558b2c044f8b20f81508bbc885be0227b2e1c6809599a89a",
    },
    "m14": {
        "sub4": "ef609e3ab5b93eaad58c73d6b2dd86752fb22dd2c8f33f24f366d7753fab29a2",
        "labels": "8bf29236b71572cc3130b71d7576840a1ab99202ee40c590484a01aeccacc2c6",
    },
    "m15": {
        "sub4": "75a6fc910332a7024ee96ca3ec394ffe75d45b71b0d6660adbb57fad4d3d3132",
        "labels": "47e6c83231153f48e709761f128f03248baffa096590675451fb75f81f6bd17e",
    },
    "m16": {
        "sub4": "3ff0201879e65573fab8573014c2293606dfaffb9132e56f622925865180999b",
        "labels": "8027e84785b307624f91d16e03c30321705fe3c5c5615275a7551a345a13a10b",
    },
    "m17": {
        "sub4": "e6e01e40cb924fbc1eb5794dc2b3d6a6e1e906212ae3113aab7dd2e3a678a26f",
        "labels": "dd32f3429312771d9f75a3072386f5d2147b5eb152f6f6b96e8106359465ff7e",
    },
    "m18": {
        "sub4": "f553f05573f6cdd86fecac1b1c214fe56695460b0b13dae12536a38028fbb747",
        "labels": "aa06f5e43e903d1c1b33506ec76ac58e12486471d7d7f69bb13289c79df1c1c3",
    },
    "m19": {
        "sub4": "6f9efe9d16387dd37c52c517ff5dc036e551a24ba444df0224035596ac1eabbd",
        "labels": "36aaf85e14255dbb7273ba110bf536fa0f8fefd3487bfc550482ad98af992e96",
    },
    "m20": {
        "sub4": "23339d32e4b0b6380ea5488e6ddddc90a501080bb605bb87c172aa22427aac9e",
        "labels": "a916b9131552e0a39e8d7a4ecd4d7996129feca9bd306f4690426e7f71509f2b",
    },
    "m21": {
        "sub4": "d64828d990ef01263aaf88e084693ce52e9011dc71e1931a68978b0f5c5dff33",
        "labels": "102a53162c20ea8c0f83be4d2fd36682e295caab40cb905cbaaeee6de0966a42",
    },
    "m22": {
        "sub4": "98142d364a38bc98fbc03eadb38d540661e52cd21fb68c32ddbff7c253111d07",
        "labels": "90319b560b018709edd2b31bc0e1f3b250a2ce9216b797c47b53544bc3983307",
    },
    "m23": {
        "sub4": "ed966b20e7cdb965c86d7d3f072c4ee601dfd6fd876b5eb63f87e910c4237ed9",
        "labels": "b37f2e9d05755f946c1714bbeab8afbfaa18da27ff0f28eb87f06d413f6b141e",
    },
    "m24": {
        "sub4": "eda4cd76e1d409e6b6da05c70e460a6b0d51858cf5c2c932c415a3ae1cc7b5fc",
        "labels": "17cc265edd1b795ac200f13caf0391880b45c656141fc93c1bed95bd53f0dae0",
    },
    "m25": {
        "sub4": "3ba9e1151ba7f0f555707160ecbc6070e027efbed51045a1a76f4190619a54ae",
        "labels": "713d7d81c7d193753465ee664970d7d6c32a7f2d0c0c1421f4a208ac109ac1c1",
    },
    "m26": {
        "sub4": "1c764582d97abdd837d3cceedd3f21d60faefbace9fa77c984dbc1821c4b757b",
        "labels": "7a9c1e8e745dc3246429a8de7e8cf48b96a9f0d74749b73bf283a9cc7a4a8972",
    },
    "m27": {
        "sub4": "64c1748e992f75caa05caca84c305ffa75d53d22e1a3f64fc55bf0eef66d810c",
        "labels": "5b5f5a977421195a913a79ab9a0997ba2284bf8d42617e28af284a487d7ed7f0",
    },
    "m28": {
        "sub4": "46cdd907a1eaa46d7992cde53de90c83b682404612919012494b6d171c676fe0",
        "labels": "e33498ebaab19ea0a845f8bab738816d309d306749f4177a5070c6d1bad51169",
    },
    "m29": {
        "sub4": "7a5d76d054cdf9e2f2c7c32cb6bfa70225ef0c1cff12ec7264650680ad11e196",
        "labels": "ca2871ba15ac74113ca931b74e7be2d27a7e716ec3f89317afa2c93703f362bb",
    },
    "m30": {
        "sub4": "8776de2ec24b142194867466150d315829825c15181181b679f6b424e0f956ec",
        "labels": "fc9c4ac97d60e5a58d3dec233a8fbf0aa9b448e5c44c6bd57c6825897bff1357",
    },
    "m31": {
        "sub4": "cb419acb89bf97b4a66f23dcd33d05d94fc8e0bb76a5a7076a8e6801c9fb5870",
        "labels": "2157a24c9bc44feabcafdf9a53af35a0313d7a49c8fd0bb209571c8760d774ae",
    },
    "m32": {
        "sub4": "72fc16c8b20944a1fc53939a6fb1df7fd15bdbe5ca5620af3419c94ce8bf8ed1",
        "labels": "7e4f1871e64fe3d6e969eae06b0f491de0e2ee03a58d28ccc255ddce45ab65ba",
    },
    "m33": {
        "sub4": "90f1a99935cc5eb4e6e1d638cf80f00241d58eb9ef3c38185b879ace3db351f3",
        "labels": "82550b9630263762e2effaedabdef15b687655f059a5c6733706cfd575cf8436",
    },
    "m34": {
        "sub4": "68a24ae5f19b79d052c23d69a84c396ec9a5e7e2430f2fc3d8aa658186003766",
        "labels": "97077748cd6d34bf4c1c5632ea34107ee744488754d76ddbe1a04fcac671f0e6",
    },
}

MHA_DTYPES = {
    "MET_UCHAR": np.uint8, "MET_CHAR": np.int8, "MET_USHORT": np.uint16, "MET_SHORT": np.int16,
    "MET_UINT": np.uint32, "MET_INT": np.int32, "MET_FLOAT": np.float32, "MET_DOUBLE": np.float64,
}


def _read_mha(path):
    """Read a MetaImage (.mha) volume with the data stored inside the file. Returns the data in zyx order."""
    header = {}
    with open(path, "rb") as f:
        while True:
            line = f.readline().decode("ascii")
            if "=" not in line:
                continue
            key, value = [s.strip() for s in line.split("=", 1)]
            header[key] = value
            if key == "ElementDataFile":
                data = f.read()
                break

    if header["ElementDataFile"] != "LOCAL":
        raise RuntimeError(f"Only .mha files with local data are supported, got '{header['ElementDataFile']}'.")
    if header.get("CompressedData", "False") == "True":
        data = zlib.decompress(data)

    dtype = np.dtype(MHA_DTYPES[header["ElementType"]])
    if header.get("BinaryDataByteOrderMSB", "False") == "True":
        dtype = dtype.newbyteorder(">")
    shape = tuple(int(s) for s in header["DimSize"].split())[::-1]  # DimSize is in xyz order.
    return np.frombuffer(data, dtype=dtype).reshape(shape)


def _convert_to_hdf5(raw_path, label_path, output_path, factor):
    import h5py
    import nrrd

    raw = _read_mha(raw_path)

    # nrrd returns the data in xyz order, we transpose it to zyx.
    labels = nrrd.read(label_path)[0].transpose(2, 1, 0)
    if factor > 1:
        # The downsampled raw volumes are block averages, so the voxel centers are shifted by (factor - 1) / 2
        # full resolution voxels. We subsample the labels with a corresponding offset and crop to the raw shape.
        offset = (factor - 1) // 2
        labels = labels[offset::factor, offset::factor, offset::factor]
        labels = labels[:raw.shape[0], :raw.shape[1], :raw.shape[2]]

    if raw.shape != labels.shape:
        raise RuntimeError(f"Shape mismatch between raw {raw.shape} and labels {labels.shape} for '{raw_path}'.")

    chunks = (1,) + raw.shape[1:]
    with h5py.File(output_path, "w") as f:
        f.create_dataset("raw", data=raw, compression="gzip", chunks=chunks)
        f.create_dataset("labels", data=labels, compression="gzip", chunks=chunks)


def get_lapd_mouse_data(
    path: Union[os.PathLike, str],
    mouse_id: str,
    resolution: Literal["full", "sub2", "sub4"] = "sub4",
    download: bool = False,
) -> str:
    """Download and preprocess one mouse of the LAPD Mouse dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        mouse_id: The mouse to download. One of the ids in `MOUSE_IDS`.
        resolution: The resolution of the raw data. Either 'full', 'sub2' (downsampled by 2) or 'sub4'
            (downsampled by 4).
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the hdf5 file with the raw data and airway labels of this mouse.
    """
    if mouse_id not in MOUSE_IDS:
        raise ValueError(f"'{mouse_id}' is not a valid mouse id. Choose one of {MOUSE_IDS}.")
    if resolution not in RESOLUTIONS:
        raise ValueError(f"'{resolution}' is not a valid resolution. Choose one of {list(RESOLUTIONS.keys())}.")

    volume_path = os.path.join(path, f"{mouse_id}_{resolution}.h5")
    if os.path.exists(volume_path):
        return volume_path

    download_dir = os.path.join(path, "downloads")
    os.makedirs(download_dir, exist_ok=True)

    suffix, factor = RESOLUTIONS[resolution]
    raw_name, label_name = f"{mouse_id}_Autofluorescent{suffix}.mha", f"{mouse_id}_AirwaySegments.nrrd"
    raw_path, label_path = os.path.join(download_dir, raw_name), os.path.join(download_dir, label_name)

    util.download_source(
        path=raw_path, url=URL.format(mouse_id=mouse_id, filename=raw_name), download=download,
        checksum=CHECKSUMS[mouse_id].get(resolution),
    )
    util.download_source(
        path=label_path, url=URL.format(mouse_id=mouse_id, filename=label_name), download=download,
        checksum=CHECKSUMS[mouse_id]["labels"],
    )

    _convert_to_hdf5(raw_path, label_path, volume_path, factor)
    os.remove(raw_path)
    os.remove(label_path)

    return volume_path


def get_lapd_mouse_paths(
    path: Union[os.PathLike, str],
    mouse_ids: Optional[Sequence[str]] = None,
    resolution: Literal["full", "sub2", "sub4"] = "sub4",
    download: bool = False,
) -> List[str]:
    """Get paths to the LAPD Mouse data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        mouse_ids: The mice to use. By default, all 34 mice are used.
        resolution: The resolution of the raw data. Either 'full', 'sub2' (downsampled by 2) or 'sub4'
            (downsampled by 4).
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the hdf5 volumes, which contain the raw data and airway labels.
    """
    if mouse_ids is None:
        mouse_ids = MOUSE_IDS
    elif isinstance(mouse_ids, str):
        mouse_ids = [mouse_ids]

    volume_paths = [get_lapd_mouse_data(path, mouse_id, resolution, download) for mouse_id in mouse_ids]
    return volume_paths


def get_lapd_mouse_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    mouse_ids: Optional[Sequence[str]] = None,
    resolution: Literal["full", "sub2", "sub4"] = "sub4",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the LAPD Mouse dataset for airway segmentation in cryomicrotome fluorescence volumes of mouse lungs.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        mouse_ids: The mice to use. By default, all 34 mice are used.
        resolution: The resolution of the raw data. Either 'full', 'sub2' (downsampled by 2) or 'sub4'
            (downsampled by 4).
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_lapd_mouse_paths(path, mouse_ids, resolution, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=volume_paths,
        raw_key="raw",
        label_paths=volume_paths,
        label_key="labels",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_lapd_mouse_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    mouse_ids: Optional[Sequence[str]] = None,
    resolution: Literal["full", "sub2", "sub4"] = "sub4",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the LAPD Mouse dataloader for airway segmentation in cryomicrotome fluorescence volumes of mouse lungs.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        mouse_ids: The mice to use. By default, all 34 mice are used.
        resolution: The resolution of the raw data. Either 'full', 'sub2' (downsampled by 2) or 'sub4'
            (downsampled by 4).
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_lapd_mouse_dataset(path, patch_shape, mouse_ids, resolution, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
