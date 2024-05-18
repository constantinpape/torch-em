import os
import hashlib
import inspect
import requests
from tqdm import tqdm
from warnings import warn
from subprocess import run
from packaging import version
from shutil import copyfileobj, which

import zipfile
import numpy as np
from xml.dom import minidom
from skimage.draw import polygon

import torch
import torch_em

try:
    import gdown
except ImportError:
    gdown = None

try:
    from tcia_utils import nbia
except ModuleNotFoundError:
    nbia = None


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


def get_bioimageio_dataset_id(dataset_name):
    assert dataset_name in BIOIMAGEIO_IDS
    return BIOIMAGEIO_IDS[dataset_name]


def get_checksum(filename):
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
def download_source(path, url, download, checksum=None, verify=True):
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


def download_source_gdrive(path, url, download, checksum=None, download_type="zip", expected_samples=10000):
    if os.path.exists(path):
        return

    if not download:
        raise RuntimeError(f"Cannot find the data at {path}, but download was set to False")

    if gdown is None:
        raise RuntimeError(
            "Need gdown library to download data from google drive."
            "Please install gdown and then rerun."
        )

    print("Downloading the dataset. Might take a few minutes...")

    if download_type == "zip":
        gdown.download(url, path, quiet=False)
        _check_checksum(path, checksum)
    elif download_type == "folder":
        assert version.parse(gdown.__version__) == version.parse("4.6.3"), "Please install `gdown==4.6.3`."
        gdown.download_folder.__globals__["MAX_NUMBER_FILES"] = expected_samples
        gdown.download_folder(url=url, output=path, quiet=True, remaining_ok=True)
    else:
        raise ValueError("`download_path` argument expects either `zip`/`folder`")
    print("Download completed.")


def download_source_empiar(path, access_id, download):
    download_path = os.path.join(path, access_id)

    if os.path.exists(download_path):
        return download_path
    if not download:
        raise RuntimeError(f"Cannot find the data at {path}, but download was set to False")

    if which("ascp") is None:
        raise RuntimeError(
            "Need aspera-cli to download data from empiar."
            "You can install it via 'mamba install -c hcc aspera-cli'."
        )

    key_file = os.path.expanduser("~/.aspera/cli/etc/asperaweb_id_dsa.openssh")
    if not os.path.exists(key_file):
        conda_root = os.environ["CONDA_PREFIX"]
        key_file = os.path.join(conda_root, "etc/asperaweb_id_dsa.openssh")

    if not os.path.exists(key_file):
        raise RuntimeError("Could not find the aspera ssh keyfile")

    cmd = [
        "ascp", "-QT", "-l", "200M", "-P33001",
        "-i", key_file, f"emp_ext2@fasp.ebi.ac.uk:/{access_id}", path
    ]
    run(cmd)

    return download_path


def download_source_kaggle(path, dataset_name, download):
    if not download:
        raise RuntimeError(f"Cannot fine the data at {path}, but download was set to False.")

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ModuleNotFoundError:
        msg = "Please install the Kaggle API. You can do this using 'pip install kaggle'. "
        msg += "After you have installed kaggle, you would need an API token. "
        msg += "Follow the instructions at https://www.kaggle.com/docs/api."
        raise ModuleNotFoundError(msg)

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset=dataset_name, path=path, quiet=False)


def download_source_tcia(path, url, dst, csv_filename, download):
    if not download:
        raise RuntimeError(f"Cannot fine the data at {path}, but download was set to False.")

    assert url.endswith(".tcia"), f"{path} is not a TCIA Manifest."

    # downloads the manifest file from the collection page
    manifest = requests.get(url=url)
    with open(path, "wb") as f:
        f.write(manifest.content)

    # this part extracts the UIDs from the manigests and downloads them.
    nbia.downloadSeries(
        series_data=path, input_type="manifest", path=dst, csv_filename=csv_filename,
    )


def update_kwargs(kwargs, key, value, msg=None):
    if key in kwargs:
        msg = f"{key} will be over-ridden in loader kwargs." if msg is None else msg
        warn(msg)
    kwargs[key] = value
    return kwargs


def unzip(zip_path, dst, remove=True):
    with zipfile.ZipFile(zip_path, "r") as f:
        f.extractall(dst)
    if remove:
        os.remove(zip_path)


def split_kwargs(function, **kwargs):
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
    if "raw_transform" not in kwargs:
        kwargs = update_kwargs(kwargs, "raw_transform", torch_em.transform.get_raw_transform())
    if "transform" not in kwargs:
        kwargs = update_kwargs(kwargs, "transform", torch_em.transform.get_augmentations(ndim=ndim))
    return kwargs


def add_instance_label_transform(
    kwargs, add_binary_target, label_dtype=None, binary=False, boundaries=False, offsets=None, binary_is_exclusive=True,
):
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


def generate_labeled_array_from_xml(shape, xml_file):
    """Function taken from: https://github.com/rshwndsz/hover-net/blob/master/lightning_hovernet.ipynb

    Given image shape and path to annotations (xml file), generatebit mask with the region inside a contour being white
        shape: The image shape on which bit mask will be made
        xml_file: path relative to the current working directory where the xml file is present

    Returns:
        An image of given shape with region inside contour being white..
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


def convert_svs_to_array(path, location=(0, 0), level=0, img_size=None):
    """Converts .svs files to numpy array format

    Argument:
        - path: [str] - Path to the svs file
        (below mentioned arguments are used for multi-resolution images)
        - location: tuple[int, int] - pixel location (x, y) in level 0 of the image (default: (0, 0))
        - level: [int] -  target level used to read the image (default: 0)
        - img_size: tuple[int, int] - expected size of the image
                                      (default: None -> obtains the original shape at the expected level)

    Returns:
        the image as numpy array

    TODO: it can be extended to convert WSIs (or modalities with multiple resolutions)
    """
    assert path.endswith(".svs"), f"The provided file ({path}) isn't in svs format"

    from tiffslide import TiffSlide

    _slide = TiffSlide(path)

    if img_size is None:
        img_size = _slide.level_dimensions[0]

    img_arr = _slide.read_region(location=location, level=level, size=img_size, as_array=True)

    return img_arr
