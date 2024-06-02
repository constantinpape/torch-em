import os
from tqdm import tqdm
from pathlib import Path
from typing import Union, Tuple

import json
import numpy as np
import imageio.v3 as imageio
from skimage.segmentation import relabel_sequential
from sklearn.model_selection import train_test_split

import torch_em

from .. import util
from ..light_microscopy.neurips_cell_seg import to_rgb


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
    if download:
        print("Download is not supported, as the data is huge and takes quite a while to download and extract.")

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


def _assort_sa_med2d_data(data_dir):
    with open(os.path.join(data_dir, "SAMed2D_v1.json")) as f:
        data = json.load(f)

    image_files = list(data.keys())

    gt_instances_dir = os.path.join(data_dir, "preprocessed_instances")
    os.makedirs(gt_instances_dir, exist_ok=True)

    skipped_files = []
    for ifile in tqdm(image_files):
        image_path = os.path.join(data_dir, ifile)
        image_id = Path(image_path).stem

        gt_path = os.path.join(gt_instances_dir, f"{image_id}.tif")
        if os.path.exists(gt_path):
            continue

        # let's split different components
        splits = image_id.split("--")
        dataset = splits[1]

        # HACK: (SKIP) there are some known images which are pretty weird (binary brain masks as inputs)
        if splits[2].find("brain-growth") != -1:
            skipped_files.append(ifile)
            continue

        # let's get the shape of the image
        image = imageio.imread(image_path)
        shape = image.shape if image.ndim == 2 else image.shape[:-1]

        # HACK: (SKIP) there are weird images which appear to be whole brain binary masks
        if dataset == "Brain_PTM":
            if len(np.unique(image)) == 2:  # easy check for binary values in the input image
                skipped_files.append(ifile)
                continue

        # let's create an empty array and merge all segmentations into one
        instances = np.zeros(shape, dtype="uint8")
        for idx, gfile in enumerate(sorted(data[ifile]), start=1):
            # HACK: (SKIP) we remove the segmentation of entire ventricular cavity in ACDC
            if dataset == "ACDC":
                if gfile.find("0003_000") != -1 and len(data[ifile]) > 1:  # to avoid whole ventricular rois
                    continue

            per_gt = imageio.imread(os.path.join(data_dir, gfile))

            # HACK: need to see if we can resize this inputs
            if per_gt.shape != shape:
                print("Skipping these images with mismatching ground-truth shapes.")
                continue

            # HACK: (UPDATE) optic disk is mapped as 0, and background as 1
            if dataset == "ichallenge_adam_task2":
                per_gt = (per_gt == 0).astype("uint8")  # simply reversing the binary optic disc masks

            instances[per_gt > 0] = idx

        instances = relabel_sequential(instances)[0]
        imageio.imwrite(gt_path, instances, compression="zlib")

    return skipped_files


def _create_splits_per_dataset(data_dir, json_file, skipped_files, val_fraction=0.1):
    with open(os.path.join(data_dir, "SAMed2D_v1.json")) as f:
        data = json.load(f)

    image_files = list(data.keys())

    breakpoint()

    # now, get's group them data-wise and make splits per dataset
    data_dict = {}
    for image_file in image_files:
        if image_file in skipped_files:
            print("Skipping this file:", image_file)
            continue

        _image_file = os.path.split(image_file)[-1]
        splits = _image_file.split("--")
        dataset = splits[1]

        if dataset in data_dict:
            data_dict[dataset].append(_image_file)
        else:
            data_dict[dataset] = [_image_file]

    # next, let's make a train-val split out of the dataset and write them in a json file
    train_dict, val_dict = {}, {}
    for dataset, dfiles in data_dict.items():
        tr_split, val_split = train_test_split(dfiles, test_size=val_fraction)
        train_dict[dataset] = tr_split
        val_dict[dataset] = val_split

    fdict = {"train": train_dict, "val": val_dict}
    with open(json_file, "w") as f:
        json.dump(fdict, f)


def _get_split_wise_paths(data_dir, json_file, split):
    with open(json_file, "r") as f:
        data = json.load(f)

    image_files = data[split]
    image_paths, gt_paths = [], []
    for dfiles in image_files.values():
        per_dataset_ipaths = [os.path.join(data_dir, "images", fname) for fname in dfiles]
        per_dataset_gpaths = [
            os.path.join(data_dir, "preprocessed_instances", f"{Path(fname).stem}.tif") for fname in dfiles
        ]

        image_paths.extend(per_dataset_ipaths)
        gt_paths.extend(per_dataset_gpaths)

    return image_paths, gt_paths


def _get_sa_med2d_paths(path, split, exclude_dataset, exclude_modality, download):
    data_dir = get_sa_med2d_data(path=path, download=download)

    json_file = os.path.join(data_dir, "preprocessed_inputs.json")
    if not os.path.exists(json_file):
        skipped_files = _assort_sa_med2d_data(data_dir=data_dir)
        _create_splits_per_dataset(data_dir=data_dir, json_file=json_file, skipped_files=skipped_files)

    image_paths, gt_paths = _get_split_wise_paths(data_dir, json_file, split)

    return image_paths, gt_paths


def get_sa_med2d_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: str,
    resize_inputs: bool = False,
    exclude_dataset: bool = None,
    exclude_modality: bool = None,
    download: bool = False,
    **kwargs
):
    """Dataset...

    You should download the dataset yourself. See `get_sa_med2d_data` for details.

    The dataset is from Ye et al. - https://doi.org/10.48550/arXiv.2311.11969.
    The dataset is curated in alignment with Cheng et al. - https://doi.org/10.48550/arXiv.2308.16184.

    Please cite it if you use it in a publication.
    """
    image_paths, gt_paths = _get_sa_med2d_paths(
        path=path, split=split, exclude_dataset=exclude_dataset, exclude_modality=exclude_modality, download=download,
    )

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs,
            patch_shape=patch_shape,
            resize_inputs=resize_inputs,
            resize_kwargs=resize_kwargs,
            ensure_rgb=to_rgb,
        )

    dataset = torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=gt_paths,
        label_key=None,
        patch_shape=patch_shape,
        ndim=2,
        with_channels=True,
        is_seg_dataset=False,
        **kwargs
    )

    return dataset


def get_sa_med2d_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    batch_size: int,
    split: str,
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
        split=split,
        resize_inputs=resize_inputs,
        exclude_dataset=exclude_dataset,
        exclude_modality=exclude_modality,
        download=download,
        **ds_kwargs
    )
    loader = torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
    return loader
