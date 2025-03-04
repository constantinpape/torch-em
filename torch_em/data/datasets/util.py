import os
import hashlib
import inspect
import zipfile
import requests
from tqdm import tqdm
from warnings import warn
from subprocess import run
from xml.dom import minidom
from packaging import version
from shutil import copyfileobj, which

from typing import Optional, Tuple, Literal

import numpy as np
from skimage.draw import polygon

import torch

import torch_em
from torch_em.transform import get_raw_transform
from torch_em.transform.generic import ResizeLongestSideInputs, Compose

try:
    import gdown
except ImportError:
    gdown = None

try:
    from tcia_utils import nbia
except ModuleNotFoundError:
    nbia = None

try:
    from cryoet_data_portal import Client, Dataset
except ImportError:
    Client, Dataset = None, None

try:
    import synapseclient
    import synapseutils
except ImportError:
    synapseclient, synapseutils = None, None


BIOIMAGEIO_IDS = {
    "covid_if": "ilastik/covid_if_training_data",
    "cremi": "ilastik/cremi_training_data",
    "dsb": "ilastik/stardist_dsb_training_data",
    "hpa": "",  # not on bioimageio yet
    "isbi2012": "ilastik/isbi2012_neuron_segmentation_challenge",
    "kasthuri": "",  # not on bioimageio yet:
    "livecell": "ilastik/livecell_dataset",
    "lucchi": "",  # not on bioimageio yet:
    "mitoem": "ilastik/mitoem_segmentation_challenge",
    "monuseg": "deepimagej/monuseg_digital_pathology_miccai2018",
    "ovules": "",  # not on bioimageio yet
    "plantseg_root": "ilastik/plantseg_root",
    "plantseg_ovules": "ilastik/plantseg_ovules",
    "platynereis": "ilastik/platynereis_em_training_data",
    "snemi": "",  # not on bioimagegio yet
    "uro_cell": "",  # not on bioimageio yet: https://doi.org/10.1016/j.compbiomed.2020.103693
    "vnc": "ilastik/vnc",
}
"""@private
"""


def get_bioimageio_dataset_id(dataset_name):
    """@private
    """
    assert dataset_name in BIOIMAGEIO_IDS
    return BIOIMAGEIO_IDS[dataset_name]


def get_checksum(filename: str) -> str:
    """Get the SHA256 checksum of a file.

    Args:
        filename: The filepath.

    Returns:
        The checksum.
    """
    with open(filename, "rb") as f:
        file_ = f.read()
        checksum = hashlib.sha256(file_).hexdigest()
    return checksum


def _check_checksum(path, checksum):
    if checksum is not None:
        this_checksum = get_checksum(path)
        if this_checksum != checksum:
            raise RuntimeError(
                "The checksum of the download does not match the expected checksum."
                f"Expected: {checksum}, got: {this_checksum}"
            )
        print("Download successful and checksums agree.")
    else:
        warn("The file was downloaded, but no checksum was provided, so the file may be corrupted.")


# this needs to be extended to support download from s3 via boto,
# if we get a resource that is available via s3 without support for http
def download_source(path: str, url: str, download: bool, checksum: Optional[str] = None, verify: bool = True) -> None:
    """Download data via https.

    Args:
        path: The path for saving the data.
        url: The url of the data.
        download: Whether to download the data if it is not saved at `path` yet.
        checksum: The expected checksum of the data.
        verify: Whether to verify the https address.
    """
    if os.path.exists(path):
        return
    if not download:
        raise RuntimeError(f"Cannot find the data at {path}, but download was set to False")

    with requests.get(url, stream=True, allow_redirects=True, verify=verify) as r:
        r.raise_for_status()  # check for error
        file_size = int(r.headers.get("Content-Length", 0))
        desc = f"Download {url} to {path}"
        if file_size == 0:
            desc += " (unknown file size)"
        with tqdm.wrapattr(r.raw, "read", total=file_size, desc=desc) as r_raw, open(path, "wb") as f:
            copyfileobj(r_raw, f)

    _check_checksum(path, checksum)


def download_source_gdrive(
    path: str,
    url: str,
    download: bool,
    checksum: Optional[str] = None,
    download_type: Literal["zip", "folder"] = "zip",
    expected_samples: int = 10000,
    quiet: bool = True,
) -> None:
    """Download data from google drive.

    Args:
        path: The path for saving the data.
        url: The url of the data.
        download: Whether to download the data if it is not saved at `path` yet.
        checksum: The expected checksum of the data.
        download_type: The download type, either 'zip' or 'folder'.
        expected_samples: The maximal number of samples in the folder.
        quiet: Whether to download quietly.
    """
    if os.path.exists(path):
        return

    if not download:
        raise RuntimeError(f"Cannot find the data at {path}, but download was set to False")

    if gdown is None:
        raise RuntimeError(
            "Need gdown library to download data from google drive. "
            "Please install gdown: 'conda install -c conda-forge gdown==4.6.3'."
        )

    print("Downloading the files. Might take a few minutes...")

    if download_type == "zip":
        gdown.download(url, path, quiet=quiet)
        _check_checksum(path, checksum)
    elif download_type == "folder":
        assert version.parse(gdown.__version__) == version.parse("4.6.3"), "Please install 'gdown==4.6.3'."
        gdown.download_folder.__globals__["MAX_NUMBER_FILES"] = expected_samples
        gdown.download_folder(url=url, output=path, quiet=quiet, remaining_ok=True)
    else:
        raise ValueError("`download_path` argument expects either `zip`/`folder`")

    print("Download completed.")


def download_source_empiar(path: str, access_id: str, download: bool) -> str:
    """Download data from EMPIAR.

    Requires the ascp command from the aspera CLI.

    Args:
        path: The path for saving the data.
        access_id: The EMPIAR accession id of the data to download.
        download: Whether to download the data if it is not saved at `path` yet.

    Returns:
        The path to the downloaded data.
    """
    download_path = os.path.join(path, access_id)

    if os.path.exists(download_path):
        return download_path
    if not download:
        raise RuntimeError(f"Cannot find the data at {path}, but download was set to False")

    if which("ascp") is None:
        raise RuntimeError(
            "Need aspera-cli to download data from empiar. You can install it via 'conda install -c hcc aspera-cli'."
        )

    key_file = os.path.expanduser("~/.aspera/cli/etc/asperaweb_id_dsa.openssh")
    if not os.path.exists(key_file):
        conda_root = os.environ["CONDA_PREFIX"]
        key_file = os.path.join(conda_root, "etc/asperaweb_id_dsa.openssh")

    if not os.path.exists(key_file):
        raise RuntimeError("Could not find the aspera ssh keyfile")

    cmd = ["ascp", "-QT", "-l", "200M", "-P33001", "-i", key_file, f"emp_ext2@fasp.ebi.ac.uk:/{access_id}", path]
    run(cmd)

    return download_path


def download_source_kaggle(path: str, dataset_name: str, download: bool, competition: bool = False):
    """Download data from Kaggle.

    Requires the Kaggle API.

    Args:
        path: The path for saving the data.
        dataset_name: The name of the dataset to download.
        download: Whether to download the data if it is not saved at `path` yet.
        competition: Whether this data is from a competition and requires the kaggle.competition API.
    """
    if not download:
        raise RuntimeError(f"Cannot find the data at {path}, but download was set to False.")

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ModuleNotFoundError:
        msg = "Please install the Kaggle API. You can do this using 'pip install kaggle'. "
        msg += "After you have installed kaggle, you would need an API token. "
        msg += "Follow the instructions at https://www.kaggle.com/docs/api."
        raise ModuleNotFoundError(msg)

    api = KaggleApi()
    api.authenticate()

    if competition:
        api.competition_download_files(competition=dataset_name, path=path, quiet=False)
    else:
        api.dataset_download_files(dataset=dataset_name, path=path, quiet=False)


def download_source_tcia(path, url, dst, csv_filename, download):
    """Download data from TCIA.

    Requires the tcia_utils python package.

    Args:
        path: The path for saving the data.
        url: The URL to the TCIA dataset.
        dst:
        csv_filename:
        download: Whether to download the data if it is not saved at `path` yet.
    """
    if nbia is None:
        raise RuntimeError("Requires the tcia_utils python package.")
    if not download:
        raise RuntimeError(f"Cannot find the data at {path}, but download was set to False.")
    assert url.endswith(".tcia"), f"{url} is not a TCIA Manifest."

    # Downloads the manifest file from the collection page.
    manifest = requests.get(url=url)
    with open(path, "wb") as f:
        f.write(manifest.content)

    # This part extracts the UIDs from the manifests and downloads them.
    nbia.downloadSeries(series_data=path, input_type="manifest", path=dst, csv_filename=csv_filename)


def download_source_synapse(path: str, entity: str, download: bool) -> None:
    """Download data from synapse.

    Requires the synapseclient python library.

    Args:
        path: The path for saving the data.
        entity: The name of the data to download from synapse.
        download: Whether to download the data if it is not saved at `path` yet.
    """
    if not download:
        raise RuntimeError(f"Cannot find the data at {path}, but download was set to False.")

    if synapseclient is None:
        raise RuntimeError(
            "You must install 'synapseclient' to download files from 'synapse'. "
            "Remember to create an account and generate an authentication code for your account. "
            "Please follow the documentation for details on creating the '~/.synapseConfig' file here: "
            "https://python-docs.synapse.org/tutorials/authentication/."
        )

    assert entity.startswith("syn"), "The entity name does not look as expected. It should be something like 'syn123'."

    # Download all files in the folder.
    syn = synapseclient.Synapse()
    syn.login()  # Since we do not pass any credentials here, it fetches all details from '~/.synapseConfig'.
    synapseutils.syncFromSynapse(syn=syn, entity=entity, path=path)


def update_kwargs(kwargs, key, value, msg=None):
    """@private
    """
    if key in kwargs:
        msg = f"{key} will be over-ridden in loader kwargs." if msg is None else msg
        warn(msg)
    kwargs[key] = value
    return kwargs


def unzip_tarfile(tar_path: str, dst: str, remove: bool = True) -> None:
    """Unpack a tar archive.

    Args:
        tar_path: Path to the tar file.
        dst: Where to unpack the archive.
        remove: Whether to remove the tar file after unpacking.
    """
    import tarfile

    if tar_path.endswith(".tar.gz") or tar_path.endswith(".tgz"):
        access_mode = "r:gz"
    elif tar_path.endswith(".tar"):
        access_mode = "r:"
    else:
        raise ValueError(f"The provided file isn't a supported archive to unpack. Please check the file: {tar_path}.")

    tar = tarfile.open(tar_path, access_mode)
    tar.extractall(dst)
    tar.close()

    if remove:
        os.remove(tar_path)


def unzip_rarfile(rar_path: str, dst: str, remove: bool = True, use_rarfile: bool = True) -> None:
    """Unpack a rar archive.

    Args:
        rar_path: Path to the rar file.
        dst: Where to unpack the archive.
        remove: Whether to remove the tar file after unpacking.
        use_rarfile: Whether to use the rarfile library or aspose.zip.
    """
    if use_rarfile:
        import rarfile
        with rarfile.RarFile(rar_path) as f:
            f.extractall(path=dst)
    else:
        import aspose.zip as az
        with az.rar.RarArchive(rar_path) as archive:
            archive.extract_to_directory(dst)

    if remove:
        os.remove(rar_path)


def unzip(zip_path: str, dst: str, remove: bool = True) -> None:
    """Unpack a zip archive.

    Args:
        zip_path: Path to the zip file.
        dst: Where to unpack the archive.
        remove: Whether to remove the tar file after unpacking.
    """
    with zipfile.ZipFile(zip_path, "r") as f:
        f.extractall(dst)
    if remove:
        os.remove(zip_path)


def split_kwargs(function, **kwargs):
    """@private
    """
    function_parameters = inspect.signature(function).parameters
    parameter_names = list(function_parameters.keys())
    other_kwargs = {k: v for k, v in kwargs.items() if k not in parameter_names}
    kwargs = {k: v for k, v in kwargs.items() if k in parameter_names}
    return kwargs, other_kwargs


# this adds the default transforms for 'raw_transform' and 'transform'
# in case these were not specified in the kwargs
# this is NOT necessary if 'default_segmentation_dataset' is used, only if a dataset class
# is used directly, e.g. in the LiveCell Loader
def ensure_transforms(ndim, **kwargs):
    """@private
    """
    if "raw_transform" not in kwargs:
        kwargs = update_kwargs(kwargs, "raw_transform", torch_em.transform.get_raw_transform())
    if "transform" not in kwargs:
        kwargs = update_kwargs(kwargs, "transform", torch_em.transform.get_augmentations(ndim=ndim))
    return kwargs


def add_instance_label_transform(
    kwargs, add_binary_target, label_dtype=None, binary=False, boundaries=False, offsets=None, binary_is_exclusive=True,
):
    """@private
    """
    if binary_is_exclusive:
        assert sum((offsets is not None, boundaries, binary)) <= 1
    else:
        assert sum((offsets is not None, boundaries)) <= 1
    if offsets is not None:
        label_transform2 = torch_em.transform.label.AffinityTransform(offsets=offsets,
                                                                      add_binary_target=add_binary_target,
                                                                      add_mask=True)
        msg = "Offsets are passed, but 'label_transform2' is in the kwargs. It will be over-ridden."
        kwargs = update_kwargs(kwargs, "label_transform2", label_transform2, msg=msg)
        label_dtype = torch.float32
    elif boundaries:
        label_transform = torch_em.transform.label.BoundaryTransform(add_binary_target=add_binary_target)
        msg = "Boundaries is set to true, but 'label_transform' is in the kwargs. It will be over-ridden."
        kwargs = update_kwargs(kwargs, "label_transform", label_transform, msg=msg)
        label_dtype = torch.float32
    elif binary:
        label_transform = torch_em.transform.label.labels_to_binary
        msg = "Binary is set to true, but 'label_transform' is in the kwargs. It will be over-ridden."
        kwargs = update_kwargs(kwargs, "label_transform", label_transform, msg=msg)
        label_dtype = torch.float32
    return kwargs, label_dtype


def update_kwargs_for_resize_trafo(kwargs, patch_shape, resize_inputs, resize_kwargs=None, ensure_rgb=None):
    """@private
    """
    # Checks for raw_transform and label_transform incoming values.
    # If yes, it will automatically merge these two transforms to apply them together.
    if resize_inputs:
        assert isinstance(resize_kwargs, dict)

        target_shape = resize_kwargs.get("patch_shape")
        if len(resize_kwargs["patch_shape"]) == 3:
            # we only need the XY dimensions to reshape the inputs along them.
            target_shape = target_shape[1:]
            # we provide the Z dimension value to return the desired number of slices and not the whole volume
            kwargs["z_ext"] = resize_kwargs["patch_shape"][0]

        raw_trafo = ResizeLongestSideInputs(target_shape=target_shape, is_rgb=resize_kwargs["is_rgb"])
        label_trafo = ResizeLongestSideInputs(target_shape=target_shape, is_label=True)

        # The patch shape provided to the dataset. Here, "None" means that the entire volume will be loaded.
        patch_shape = None

    if ensure_rgb is None:
        raw_trafos = []
    else:
        assert not isinstance(ensure_rgb, bool), "'ensure_rgb' is expected to be a function."
        raw_trafos = [ensure_rgb]

    if "raw_transform" in kwargs:
        raw_trafos.extend([raw_trafo, kwargs["raw_transform"]])
    else:
        raw_trafos.extend([raw_trafo, get_raw_transform()])

    kwargs["raw_transform"] = Compose(*raw_trafos, is_multi_tensor=False)

    if "label_transform" in kwargs:
        trafo = Compose(label_trafo, kwargs["label_transform"], is_multi_tensor=False)
        kwargs["label_transform"] = trafo
    else:
        kwargs["label_transform"] = label_trafo

    return kwargs, patch_shape


def generate_labeled_array_from_xml(shape: Tuple[int, ...], xml_file: str) -> np.ndarray:
    """Generate a label mask from a contour defined in a xml annotation file.

    Function taken from: https://github.com/rshwndsz/hover-net/blob/master/lightning_hovernet.ipynb

    Args:
        shape: The image shape.
        xml_file: The path to the xml file with contour annotations.

    Returns:
        The label mask.
    """
    # DOM object created by the minidom parser
    xDoc = minidom.parse(xml_file)

    # List of all Region tags
    regions = xDoc.getElementsByTagName('Region')

    # List which will store the vertices for each region
    xy = []
    for region in regions:
        # Loading all the vertices in the region
        vertices = region.getElementsByTagName('Vertex')

        # The vertices of a region will be stored in a array
        vw = np.zeros((len(vertices), 2))

        for index, vertex in enumerate(vertices):
            # Storing the values of x and y coordinate after conversion
            vw[index][0] = float(vertex.getAttribute('X'))
            vw[index][1] = float(vertex.getAttribute('Y'))

        # Append the vertices of a region
        xy.append(np.int32(vw))

    # Creating a completely black image
    mask = np.zeros(shape, np.float32)

    for i, contour in enumerate(xy):
        r, c = polygon(np.array(contour)[:, 1], np.array(contour)[:, 0], shape=shape)
        mask[r, c] = i
    return mask


# This function could be extended to convert WSIs (or modalities with multiple resolutions).
def convert_svs_to_array(
    path: str, location: Tuple[int, int] = (0, 0), level: int = 0, img_size: Tuple[int, int] = None,
) -> np.ndarray:
    """Convert a .svs file for WSI imagging to a numpy array.

    Requires the tiffslide python library.
    The function can load multi-resolution images. You can specify the resolution level via `level`.

    Args:
        path: File path ath to the svs file.
        location: Pixel location (x, y) in level 0 of the image.
        level: Target level used to read the image.
        img_size: Size of the image. If None, the shape of the image at `level` is used.

    Returns:
        The image as numpy array.
    """
    from tiffslide import TiffSlide

    assert path.endswith(".svs"), f"The provided file ({path}) isn't in svs format"
    _slide = TiffSlide(path)
    if img_size is None:
        img_size = _slide.level_dimensions[0]
    return _slide.read_region(location=location, level=level, size=img_size, as_array=True)


def download_from_cryo_et_portal(path: str, dataset_id: int, download: bool) -> str:
    """Download data from the CryoET Data Portal.

    Requires the cryoet-data-portal python library.

    Args:
        path: The path for saving the data.
        dataset_id: The id of the data to download from the portal.
        download: Whether to download the data if it is not saved at `path` yet.

    Returns:
        The file path to the downloaded data.
    """
    if Client is None or Dataset is None:
        raise RuntimeError("Please install CryoETDataPortal via 'pip install cryoet-data-portal'")

    output_path = os.path.join(path, str(dataset_id))
    if os.path.exists(output_path):
        return output_path

    if not download:
        raise RuntimeError(f"Cannot find the data at {path}, but download was set to False.")

    client = Client()
    dataset = Dataset.get_by_id(client, dataset_id)
    dataset.download_everything(dest_path=path)

    return output_path
