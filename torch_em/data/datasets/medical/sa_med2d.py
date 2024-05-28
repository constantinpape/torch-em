import os
from typing import Union, Tuple

import torch_em

from .. import util


DATASET_NAMES = [
    "ACDC",
]

MODALITY_NAMES = [
    "ct",
]


def get_sa_med2d_data(path, download):
    """This function describes the download functionality and ensures your data has been downloaded in expected format.

    The dataset is located at https://huggingface.co/datasets/OpenGVLab/SA-Med2D-20M.

    There are two ways of downloading the dataset:
    1. wget (Recommended):
        - There are 10 `z.*` files and 1 `.zip` file which needs to be installed together.
        - Go to `Files` -> download each file individually using `wget <LINK>`. Below mentioned are the links:
            - https://huggingface.co/datasets/OpenGVLab/SA-Med2D-20M/resolve/main/raw/SA-Med2D-16M.z01
            - https://huggingface.co/datasets/OpenGVLab/SA-Med2D-20M/resolve/main/raw/SA-Med2D-16M.z02
            - https://huggingface.co/datasets/OpenGVLab/SA-Med2D-20M/resolve/main/raw/SA-Med2D-16M.z03
            - https://huggingface.co/datasets/OpenGVLab/SA-Med2D-20M/resolve/main/raw/SA-Med2D-16M.z04
            - https://huggingface.co/datasets/OpenGVLab/SA-Med2D-20M/resolve/main/raw/SA-Med2D-16M.z05
            - https://huggingface.co/datasets/OpenGVLab/SA-Med2D-20M/resolve/main/raw/SA-Med2D-16M.z06
            - https://huggingface.co/datasets/OpenGVLab/SA-Med2D-20M/resolve/main/raw/SA-Med2D-16M.z07
            - https://huggingface.co/datasets/OpenGVLab/SA-Med2D-20M/resolve/main/raw/SA-Med2D-16M.z08
            - https://huggingface.co/datasets/OpenGVLab/SA-Med2D-20M/resolve/main/raw/SA-Med2D-16M.z09
            - https://huggingface.co/datasets/OpenGVLab/SA-Med2D-20M/resolve/main/raw/SA-Med2D-16M.z10
            - https://huggingface.co/datasets/OpenGVLab/SA-Med2D-20M/resolve/main/raw/SA-Med2D-16M.zip

    2. Using Git Large File Storage (lfs):
        - `git lfs install` (Make sure you have git-lfs installed (https://git-lfs.com))
        - `git clone https://huggingface.co/datasets/OpenGVLab/SA-Med2D-20M`
            - This step takes several hours, make sure you have a consistent internet and sufficient space.

    Once you have downloaded the archives, you need to unzip the splitted-up zip files:
    - For Windows: decompress SA-Med2D-16M.zip to automatically extract the other volumes together.
    - For Linux:
        - `zip SA-Med2D-16M.zip SA-Med2D-16M.z0* SA-Med2D-16M.z10 -s=0 --out {full}.zip`
            - NOTE: deflates the entire dataset to ensemble into one zip, make sure you have ~1.5TB free space.
        - `unzip {full}.zip`
            - NOTE: there are >4M images paired with >19M ground-truth masks. unzipping takes a lot of inodes and time.
    """
    data_dir = os.path.join(path, "SAMed2Dv1")

    # the first part is to ensure if the data has been unzipped in the expected data directory
    msg = "The data directory is not found. "
    msg += "Please ensure that you provide the path to the parent directory where the unzip operation took place. "
    msg += "For example: `unzip <ZIPFILE> -d /path/to/dir/`. Hence, the argument 'path' expects '/path/to/dir/'."
    assert os.path.exists(data_dir), msg

    # next, let's investigate the presence of the json files
    json_file = "SAMed2D_v1.json"
    assert os.path.exists(os.path.join(data_dir, json_file)), f"The json file '{json_file}' is missing."

    json_file = "SAMed2D_v1_class_mapping_id.json"
    assert os.path.exists(os.path.join(data_dir, json_file)), f"The json file '{json_file}' is missing."

    print("Looks like the dataset is ready to use.")

    return data_dir


def _get_sa_med2d_paths(path, exclude_dataset, exclude_modality, download):
    data_dir = get_sa_med2d_data(path=path, download=download)

    image_paths = ...
    gt_paths = ...

    return image_paths, gt_paths


def get_sa_med2d_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    exclude_dataset: bool = None,
    exclude_modality: bool = None,
    download: bool = False,
    **kwargs
):
    """Dataset...

    The dataset is from Ye et al. - https://doi.org/10.48550/arXiv.2311.11969.
    The dataset is curated in alignment with Cheng et al. - https://doi.org/10.48550/arXiv.2308.16184.

    Please cite it if you use it in a publication.
    """
    image_paths, gt_paths = _get_sa_med2d_paths(
        path=path, exclude_dataset=exclude_dataset, exclude_modality=exclude_modality, download=download,
    )

    dataset = torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=gt_paths,
        label_key=None,
        patch_shape=patch_shape,
        **kwargs
    )

    return dataset


def get_sa_med2d_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    batch_size: int,
    resize_inputs: bool = False,
    exclude_dataset: bool = None,
    exclude_modality: bool = None,
    download: bool = False,
    **kwargs
):
    """Dataloader...
    See `get_sa_med2d_dataset` for details.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_sa_med2d_dataset(
        path=path,
        patch_shape=patch_shape,
        resize_inputs=resize_inputs,
        exclude_dataset=exclude_dataset,
        exclude_modality=exclude_modality,
        download=download,
        **ds_kwargs
    )
    loader = torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
    return loader
