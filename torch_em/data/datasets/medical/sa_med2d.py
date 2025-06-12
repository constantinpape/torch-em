"""The SA-Med2D-20M dataset contains annotations for several organs and structures in biomedical
images from several imaging modalities.

NOTE: The current version contains 3.7M images and 15.8M masks.

The dataset is located in HuggingFace at https://huggingface.co/datasets/OpenGVLab/SA-Med2D-20M.
The dataset is from the publication: https://arxiv.org/abs/2311.11969.
And the dataset is curated in alignment with the publication: https://doi.org/10.48550/arXiv.2308.16184.
Please cite it if you use this dataset in your research.
"""

import os
import shutil
import zipfile
from glob import glob
from math import ceil
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import json
import numpy as np
import imageio.v3 as imageio
from skimage.segmentation import relabel_sequential

from torch.utils.data import Dataset, DataLoader

import torch_em
from torch_em.transform.generic import ResizeLongestSideInputs

from .. import util


DATASET_NAMES = [
    "ACDC",  # cardiac structures in MRI
    "AMOS2022",  # multi-organ in CT
    "ATM2022",  # pulmonary airway in CT
    "AbdomenCT1K",  # abdominal organ in CT
    "ASC18",  # left atrium in LGE-MRI
    "COSMOS2022",  # cartoid vessel wall in MRI
    "BTCV",  # organs in CT
    "BTCV_Cervix",  # cervical organs in CT
    "BraTS2013",  # brain tumour in MRI
    "BraTS2015",  # brain tumour in MRI
    "BraTS2018",  # brain tumour in MRI
    "BraTS2019",  # brain tumour in MRI
    "BraTS2020",  # brain tumour in MRI
    "BraTS2021",  # brain tumour in MRI
    "Brain_PTM",  # white matter tracts in brain MRI
    "CAD_PE",  # pulmonary embolism in CTPA
    "CHAOS_Task_4",  # liver, kidney and spleen in T1W-MR
    "CMRxMotions",  # cardiac structures in CMR
    "COVID19CTscans",  # lung and covid infection in CT
    "COVID-19-20",  # covid infection in CT
    "covid_19_ct_cxr",  # lung in CXR
    "crass",  # clavicle in CXR
    "CTPelvic1k",  # pelvic bones in CT
    "CTSpine1K_Full",  # spinal vertebrae in CT
    "cvc_clinicdb",  # polyp in colonoscopy
    "Chest_Image_Pneum",  # pneumonia in CXR
    "cranium",  # cranial segmentation in CT
    "CrossMoDA21",  # vestibular schwannoma and cochlea segmentation in T1-CE and TI-HR MRI
    "CrossMoDA22",  # vestibular schwannoma and cochlea segmentation in T1-CE and TI-HR MRI
    "EMIDEC",  # cardiac structures in MRI
    "endovis15",  # polyp in endoscopy
    "FLARE21",  # abdominal organs in CT
    "FLARE22",  # abdominal organs in CT
    "fusc2021",  # skin lesion in dermoscopy
    "hvsmr_2016",  # blood pool and ventricular myocardium in CMR
    "Heart_Seg_MRI",  # heart in MRI
    "ichallenge_adam_task2",  # optic disc in fundus images
    "PALM19",  # optic disc in fundus images
    "gamma",  # optic disk, optic cup and ring in fundus images
    "gamma3",  # optic disk, optic cup and ring in fundus images
    "ISLES_SPES",  # ischemic stroke lesion in brain MRI
    "ISLES_SISS",  # ischemic stroke lesion in brain MRI
    "ISLES2016",  # ischemic stroke lesion in brain MRI
    "ISLES2017",  # ischemic stroke lesion in brain MRI
    "ISLES2018",  # ischemic stroke in brain CT
    "ISLES2022",  # ischemic stroke in brain MRI
    "Instance22",  # intracranial hemorrhage in nc-ct
    "KiTS",  # kidney and kidney tumor in CT
    "KiTS2021",  # kidney and kidney tumor in CT
    "LNDb",  # lung nodules in thoracic CT
    "LUNA16",  # lung and trachea in thoracic CT
    "LongitudinalMultipleSclerosisLesionSegmentation",  # MS lesion in FLAIR-MRI
    "mnms2",  # cardiac structures in MRI
    "MMWHS",  # whole heart in CT
    "BrainTumour",  # brain tumor in MRI
    "MSD_Heart",  # heart in MRI
    "MSD_Liver",  # liver in CT
    "MSD_Prostate",  # prostate in ADC-MRI
    "MSD_Lung",  # lung tumour in CT
    "MSD_Pancreas",  # pancreas in CT
    "MSD_HepaticVessel",  # hepatic vessel in CT
    "MSD_Spleen",  # spleen in CT
    "MSD_Colon",  # colon in CT
    "CT_ORG",  # multiple organ in CT
    "picai_baseline",  # prostate cancer in MRI
    "picai_semi",  # prostate cancer in MRI
    "Promise09",  # prostate in MRI
    "PROMISE12",  # prostate in MRI
    "Parse22",  # pulmonary atery in CT
    "chest_x_ray_images_with_pneumothorax_masks",  # pneumothorax in CXR
    "Prostate_MRI_Segmentation_Dataset",  # prostate in MRI
    "Pulmonary_Chest_X-Ray_Abnormalities_seg",  # lung in CXR
    "QUBIQ2020",  # kidney in CT
    "StructSeg2019_subtask1",  # OAR in H&N CT
    "StructSeg2019_subtask2",  # OAR in chest CT
    "Totalsegmentator_dataset",  # organ in CT
    "ultrasound_nerve_segmentation",  # nerve in US
    "VESSEL2012",  # lung in CT
    "VerSe20",  # vertebrae in CT
    "VerSe19",  # vertebrae in CT
    "WORD",  # abdominal organs in CT
    "autoPET",  # lesions in PET and CT
    "braimMRI",  # brain lesions in MRI
    "breast_ultrasound_images_dataset",  # breast cancer in US
    "kvasircapsule_seg",  # polyp in endoscopy
    "sz_cxr",  # lungs in CXR
    "EndoVis_2017_RIS",  # instruments in endoscopy
    "kvasir_seg",  # polyp in endoscopy
    "isic2018_task1",  # skin lesions in dermoscopy
    "isic2017_task1",  # skin lesions in dermoscopy
    "isic2016_task1",  # skin lesions in dermoscopy
]

MODALITY_NAMES = [
    # CT modalities
    'ct_00', 'ct_cbf', 'ct_cbv', 'ct_mtt', 'ct_tmax',
    # RGB0-image modalities
    'dermoscopy_00', 'endoscopy_00', 'fundus_photography',
    # MRI modalities
    'mr_00', 'mr_adc', 'mr_cbf', 'mr_cbv', 'mr_cmr', 'mr_dwi',
    'mr_flair', 'mr_hbv', 'mr_lge', 'mr_mprage', 'mr_mtt',
    'mr_pd', 'mr_rcbf', 'mr_rcbv', 'mr_t1', 'mr_t1c', 'mr_t1ce',
    'mr_t1gd', 'mr_t1w', 'mr_t2', 'mr_t2w', 'mr_tmax', 'mr_ttp',
    # mono-channel modalities
    'pet_00', 'ultrasound_00', 'x_ray'
]


# datasets under 1000 samples
SMALL_DATASETS = [
    "crass", "covid_19_ct_cxr", "cvc_clinicdb", "cranium", "CrossMoDA21", "EMIDEC",
    "endovis15", "fusc2021", "Heart_Seg_MRI", "ichallenge_adam_task2", "gamma", "gamma3",
    "Instance22", "LNDb", "MSD_Heart", "MSD_Prostate", "MSD_Spleen", "MSD_Colon",
    "picai_baseline", "picai_semi", "Promise09", "PROMISE12", "Pulmonary_Chest_X-Ray_Abnormalities_seg",
    "QUBIQ2020", "breast_ultrasound_images_dataset", "kvasircapsule_seg", "sz_cxr", "kvasir_seg"
]

SHARD_SIZE = 50000   # maximum images per dataset container file.


def _preprocess_data(path):
    import h5py

    data_dir = os.path.join(path, "data")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(data_dir, exist_ok=True)

    # We must ensure that the core zipfile (all small zipped splits merged into one) exists as expected.
    zip_path = os.path.join(path, "data.zip")  # NOTE: The zipfile name is hard-coded to 'data.zip'.
    if not os.path.exists(zip_path):
        raise FileNotFoundError(
            f"The combined zip file does not exist under the file name 'data.zip' at '{path}'. "
            "Please see 'get_sa_med2d_data' for details."
        )

    # Function to preprocess each image.
    def _process_each_image(image_path, data, dataset_name, data_dir, raw_transform, label_transform):
        image = imageio.imread(image_path)

        if image.ndim == 3:
            image = image.transpose(2, 0, 1)  # Make channels first for the transform to work.
        else:
            assert image.ndim == 2, image.ndim
            image = np.stack([image] * 3, axis=0)

        shape = image.shape[1:]

        # Get the image filename.
        image_fname = f"images/{os.path.basename(image_path)}"
        instances = np.zeros(shape, dtype="uint8")

        # Merge all masks into one label image.
        for idx, gt_fname in enumerate(sorted(data.get(image_fname, [])), start=1):
            # HACK: (SKIP) We remove the segmentation of entire ventricular cavity in ACDC.
            # Avoid whole ventricular rois specifically.
            if dataset_name == "ACDC" and "0003_000" in gt_fname and len(data[image_fname]) > 1:
                continue

            gt_path = os.path.join(data_dir, "SAMed2Dv1", gt_fname)
            gt_mask = imageio.imread(gt_path)

            if gt_mask.shape != shape:
                print("Skipping these images with mismatching ground-truth shapes.")
                continue

            # HACK: (UPDATE) The optic disk is mapped as 0, and background as 1
            if dataset_name == "ichallenge_adam_task2":
                gt_mask = (gt_mask == 0).astype("uint8")  # Simply reversing binary optic disc masks.

            instances[gt_mask > 0] = idx

        # Check if the image and corresponding labels are valid.
        if len(np.unique(instances)) > 1 and len(np.unique(image)) > 1:
            # This checks if the label has atleast one foreground object and the raw data has some valid information.
            instances = relabel_sequential(instances)[0]
            return raw_transform(image), label_transform(instances)
        else:
            return None

    print("We will start pre-processing the dataset. This might take a while.")
    with zipfile.ZipFile(zip_path, "r") as f:
        all_members = f.namelist()

        # First, we extract json files.
        json_members = [m for m in all_members if m.endswith(".json")]
        f.extractall(path=data_dir, members=json_members)

        # Load the json file.
        with open(os.path.join(data_dir, "SAMed2Dv1", "SAMed2D_v1.json")) as j:
            data = json.load(j)

        # Get image and label transforms to resize images to expected patch shape for training.
        raw_transform = ResizeLongestSideInputs(target_shape=(512, 512), is_rgb=True)
        label_transform = ResizeLongestSideInputs(target_shape=(512, 512), is_label=True)

        # Get members per dataset and extract them one-by-one.
        for dataset_name in tqdm(DATASET_NAMES, desc="Preprocessing data"):
            # First, we check if this dataset has any related h5 files, otherwise proceed with extraction.
            if len(glob(os.path.join(data_dir, f"{dataset_name}*.h5"))) > 0:
                continue

            # Extract only the images and labels matching the dataset name.
            dataset_members = [m for m in all_members if dataset_name in m]
            f.extractall(path=data_dir, members=dataset_members)

            # Get all image and label paths.
            image_dir = os.path.join(data_dir, "SAMed2Dv1", "images")
            image_paths = natsorted(glob(os.path.join(image_dir, "*")))
            num_images = len(image_paths)

            # Compute the total number of shards.
            # The files blow up some strange buffer memory, so I just piece the datasets down a bit.
            num_shards = ceil(num_images / SHARD_SIZE)

            for shard_idx in range(num_shards):
                start_idx = shard_idx * SHARD_SIZE
                end_idx = min((shard_idx + 1) * SHARD_SIZE, num_images)
                shard_image_paths = image_paths[start_idx:end_idx]

                # Store all images in current set inside one h5 file.
                shard_fpath = os.path.join(data_dir, f"{dataset_name}_{shard_idx:02d}.h5")
                if os.path.exists(shard_fpath):
                    continue

                with h5py.File(shard_fpath, "w") as h:
                    raw_ds = h.create_dataset(
                        "raw",
                        shape=(3, 0, 512, 512),
                        maxshape=(3, None, 512, 512),
                        chunks=(3, 1, 512, 512),
                        compression="lzf",
                    )
                    label_ds = h.create_dataset(
                        "labels",
                        shape=(0, 512, 512),
                        maxshape=(None, 512, 512),
                        chunks=(1, 512, 512),
                        compression="lzf",
                    )

                    # We need to preprocess images and corresponding labels, and store them.
                    curr_len = 0
                    with ThreadPoolExecutor(max_workers=32) as executor:
                        futures = [
                            executor.submit(
                                _process_each_image,
                                image_path, data, dataset_name, data_dir, raw_transform, label_transform,
                            ) for image_path in shard_image_paths
                        ]

                        for i, future in enumerate(
                            tqdm(
                                as_completed(futures), total=len(futures),
                                desc=f"Processing '{dataset_name}' images for shard '{shard_idx:02d}'")
                        ):
                            result = future.result()

                            if result is None:  # When the image or corresponding labels are not valid.
                                print(f"Skipping invalid image and labels: {shard_image_paths[i]}")
                                continue

                            image_transformed, label_transformed = result

                            # We resize the dataset object to incrementally add new samples.
                            raw_ds.resize((3, curr_len + 1, 512, 512))
                            label_ds.resize((curr_len + 1, 512, 512))

                            # Let's write the images and labels incrementally.
                            raw_ds[:, curr_len] = image_transformed
                            label_ds[curr_len] = label_transformed

                            curr_len += 1

            # And finally, remove all files for the current dataset at the end.
            shutil.rmtree(os.path.join(data_dir, "SAMed2Dv1", "images"))
            shutil.rmtree(os.path.join(data_dir, "SAMed2Dv1", "masks"))

    # And remove the json files as well
    shutil.rmtree(os.path.join(data_dir, "SAMed2Dv1"))

    return data_dir


def get_sa_med2d_data(path: Union[os.PathLike, str], download: bool = False) -> str:
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

    Once you have downloaded the archives, please run the following script to create one unified zipfile:
    - zip SA-Med2D-16M.zip SA-Med2D-16M.z0* SA-Med2D-16M.z10 -s=0 --out data.zip`
        - NOTE: deflates the entire dataset to ensemble into one zip, make sure you have ~1.5TB free space.

    And the following preprocessing parts are taken care of by `get_sa_med2d_data` for you.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is already downloaded and unzipped.
    """
    if download:
        print("Download is not supported, as the data is huge and takes quite a while to download and extract.")

    # And the final stage is preprocessing the images to be able to efficiently access the entire dataset.
    data_dir = _preprocess_data(path)
    print("Looks like the dataset is ready to use.")
    return data_dir


def get_sa_med2d_paths(path: Union[os.PathLike, str], download: bool = False) -> List[str]:
    """Get paths to the SA-Med2D-20M data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the input data.
    """
    data_dir = get_sa_med2d_data(path, download)
    input_paths = natsorted(glob(os.path.join(data_dir, "*.h5")))
    return input_paths


def get_sa_med2d_dataset(
    path: Union[os.PathLike, str], patch_shape: Tuple[int, int], download: bool = False, **kwargs,
) -> Dataset:
    """Get the SA-Med2D-20M dataset for various medical image segmentation tasks.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    input_paths = get_sa_med2d_paths(path, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=input_paths,
        raw_key="raw",
        label_paths=input_paths,
        label_key="labels",
        patch_shape=patch_shape,
        ndim=2,
        with_channels=True,
        is_seg_dataset=True,
        verify_paths=False,
        **kwargs
    )


def get_sa_med2d_loader(
    path: Union[os.PathLike, str], batch_size: int, patch_shape: Tuple[int, int], download: bool = False, **kwargs,
) -> DataLoader:
    """Get the SA-Med2D-20M dataloader for various medical image segmentation tasks.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_sa_med2d_dataset(path, patch_shape, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
