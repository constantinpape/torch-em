"""The StructSeg dataset contains annotations for organ-at-risk (OAR) and gross target volume (GTV)
segmentation in head and neck and in chest CT scans of nasopharynx and lung cancer patients.

It comprises the training sets of the four tasks of the StructSeg2019 challenge
(https://structseg2019.grand-challenge.org), each with 50 annotated CT volumes:
- task1 ('HaN_OAR'): 22 organs at risk in head and neck CT scans of nasopharynx cancer patients.
- task2 ('Naso_GTV'): the gross target volume of nasopharynx cancer, on the same 50 scans as task1.
- task3 ('Thoracic_OAR'): 6 organs at risk in chest CT scans of lung cancer patients.
- task4 ('Lung_GTV'): the gross target volume of lung cancer, on the same 50 scans as task3.
The 10 test volumes per task are distributed without annotations and are therefore not included here.

NOTE: The label legend is as follows (see `LABEL_IDS`):
- task1: background: 0, brainstem: 1, left eye: 2, right eye: 3, left lens: 4, right lens: 5,
  left optic nerve: 6, right optic nerve: 7, optic chiasm: 8, left temporal lobe: 9,
  right temporal lobe: 10, pituitary: 11, left parotid gland: 12, right parotid gland: 13,
  left inner ear: 14, right inner ear: 15, left middle ear: 16, right middle ear: 17,
  left temporomandibular joint: 18, right temporomandibular joint: 19, spinal cord: 20,
  left mandible: 21, right mandible: 22
- task2: background: 0, nasopharynx cancer gross target volume: 1
- task3: background: 0, left lung: 1, right lung: 2, heart: 3, esophagus: 4, trachea: 5, spinal cord: 6
- task4: background: 0, lung cancer gross target volume: 1
The id order is taken from https://github.com/zhilothebest/Coronary_Calcium, which transcribes the readme of
the official release: task1 lists 'Brain_Stem, Eye_L, Eye_R, Lens_L, Lens_R, Opt_Nerve_L, Opt_Nerve_R,
Opt_Chiasma, Temporal_Lobes_L, Temporal_Lobes_R, Pituitary, Parotid_Gland_L, Parotid_Gland_R, Inner_Ear_L,
Inner_Ear_R, Mid_Ear_L, Mid_Ear_R, T_M_Joint_L, T_M_Joint_R, Spinal_Cord, Mandible_L, Mandible_R' and task3
lists 'Lung_L, Lung_R, Heart, Esophagus, Trachea, Spinal_Cord', each 'corresponding to the label 1 to N in
the annotation file'. NOTE: Neither order is the order in which the organs are listed on
https://structseg2019.grand-challenge.org/Dataset/, and the per-label statistics published in
https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/StructSeg2019_Task3.md carry the
task3 names in the website order instead of the id order, so they call id 3 the spinal cord when it is in
fact the heart.

The task1 ids were verified on the data: all 50 volumes contain exactly the ids 0 - 22, the mean intensity
identifies the bony structures (inner ear 1043 HU, mandible 734 HU, temporomandibular joint 408 HU) and the
soft tissue ones (eye 25 HU, brainstem 29 HU, parotid gland 22 HU), the mean volumes match the expected
anatomy (lens 0.27 cm3, pituitary 0.58 cm3, optic chiasm 1.05 cm3, eye 9.1 cm3, brainstem 27 cm3,
temporal lobe 106 cm3), and in the LPS orientation of the volumes every structure named '_l' lies on the
patient's left of its '_r' partner.

The task3 ids 1 - 4 were confirmed independently of that transcription, against the copy of the task3 volumes
redistributed in https://huggingface.co/datasets/blueyo0/SA-Med3D-140K (its 'ct_general_StructSeg2019_subtask2'
entries), which stores one named binary mask per organ. Over the 50 cases the volumes of its 'lung_left',
'lung_right', 'heart' and 'esophagus' masks match the statistics published for the ids 1, 2, 3 and 4 to within
0.3% (median 1324 / 1842 / 690 / 35.7 cm3 against 1325 / 1841 / 692 / 35.5 cm3, and the minima and maxima agree
just as closely). The ids 5 and 6 could not be checked that way, since that copy holds neither a trachea nor a
spinal cord mask, so they rest on the transcribed readme alone.

The data of 'task1' is downloaded from the redistribution at
https://huggingface.co/datasets/Luffy503/VoCo_Downstream (the file 'Dataset190_Structseg19.zip'), which is
published as part of the VoCo benchmark (https://github.com/Luffy03/Large-Scale-Medical) and is tagged
Apache-2.0 by its uploader. NOTE: This is a third-party redistribution of data that the official release
only hands out to registered challenge participants, and the StructSeg organizers did not grant a license
for it. The participant agreement is quoted in https://github.com/zhilothebest/Coronary_Calcium as saying
that 'the datasets can not be publicly posted, or distributed to anyone outside of this project', so please
make sure that you are allowed to use the data for your purpose. The redistribution
stores the data in the nnU-Net layout, which this module converts to the layout of the official release.
Its content was verified against the official release: 50 CT volumes, the label ids 0 - 22, and the same
file sizes as the copy of 'Task1_HaN_OAR.zip' that is linked from
https://github.com/openmedlab/MedLSAM.

NOTE: There is no openly published copy of 'task2', 'task3' and 'task4'. They require registration and
cannot be downloaded automatically. Please follow these steps:
- Register for the challenge at https://structseg2019.grand-challenge.org and join it, then visit
  https://structseg2019.grand-challenge.org/Download/ and follow the instructions of the organizers to
  obtain the archives 'Task2_Naso_GTV.zip', 'Task3_Thoracic_OAR.zip' and 'Task4_Lung_GTV.zip' (the GTV
  tasks are also deposited at https://doi.org/10.21227/h75x-gt46, which needs a paid IEEE DataPort
  subscription and holds only the 83 byte file 'download url.txt' with a download url, not the images).
- Extract the archives of the tasks you need into '<path>', such that
  '<path>/Naso_GTV/<case_id>/data.nii.gz' and '<path>/Naso_GTV/<case_id>/label.nii.gz' exist
  (and equivalently '<path>/Thoracic_OAR/...' and '<path>/Lung_GTV/...').
  The case ids are the numbers 1 - 50. An extra enclosing folder, e.g.
  '<path>/Task3_Thoracic_OAR/Thoracic_OAR/1/data.nii.gz', is handled as well. A manual download of
  'Task1_HaN_OAR.zip' is picked up in the same way and is then used instead of the redistribution.
The only openly published parts of these three tasks that could be found are in
https://huggingface.co/datasets/blueyo0/SA-Med3D-140K, which is not a usable substitute: it holds the 50
task3 volumes with masks for only 4 of the 6 organs (no spinal cord and no trachea), the 50 task2 volumes
without any labels, and nothing at all of task4, all resampled to an isotropic spacing of 1.5 mm. Note that
its task numbering differs from the challenge: its 'subtask2' entries are task3 and its 'subtask4' entries
are task2.

The most promising lead for the two GTV tasks is https://rec.ustc.edu.cn/share/b812d430-f577-11ed-a202-03afc6a1d18f,
which https://github.com/shijun18/GTV_AutoSeg offers as the code and data of a paper that evaluates on exactly
task2 and task4, and whose author also made the IEEE DataPort deposit above, so it is probably the url that
the deposit points to. It is not open either: listing that share needs a login at the hosting site.

The dataset is located at https://structseg2019.grand-challenge.org.

The challenge does not have a dedicated publication. Please cite the challenge website and, for the
gross target volume tasks, the data record https://doi.org/10.21227/h75x-gt46 if you use this dataset
in your research.
"""

import os
import shutil
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


TASK_FOLDERS = {
    "task1": "HaN_OAR", "task2": "Naso_GTV", "task3": "Thoracic_OAR", "task4": "Lung_GTV",
}

# Only 'task1' has an openly published copy, see the module docstring.
URLS = {"task1": "https://huggingface.co/datasets/Luffy503/VoCo_Downstream/resolve/main/Dataset190_Structseg19.zip"}

CHECKSUMS = {"task1": "0e818cba20bbb783512c6062abc8e2adb43545f5870d630128df1912006c7703"}

MIRROR_FOLDERS = {"task1": "Dataset190_Structseg19"}

HAN_OAR_NAMES = [
    "brainstem", "eye_l", "eye_r", "lens_l", "lens_r", "optic_nerve_l", "optic_nerve_r", "optic_chiasm",
    "temporal_lobe_l", "temporal_lobe_r", "pituitary", "parotid_gland_l", "parotid_gland_r", "inner_ear_l",
    "inner_ear_r", "middle_ear_l", "middle_ear_r", "tm_joint_l", "tm_joint_r", "spinal_cord", "mandible_l",
    "mandible_r",
]

THORACIC_OAR_NAMES = ["lung_l", "lung_r", "heart", "esophagus", "trachea", "spinal_cord"]

LABEL_IDS = {
    "task1": {"background": 0, **{name: i + 1 for i, name in enumerate(HAN_OAR_NAMES)}},
    "task2": {"background": 0, "nasopharynx_gtv": 1},
    "task3": {"background": 0, **{name: i + 1 for i, name in enumerate(THORACIC_OAR_NAMES)}},
    "task4": {"background": 0, "lung_gtv": 1},
}


def _find_task_dir(path, task):
    folder = TASK_FOLDERS[task]
    candidates = [os.path.join(path, folder), *glob(os.path.join(path, "*", folder))]
    for candidate in candidates:
        if len(glob(os.path.join(candidate, "*", "data.nii.gz"))) > 0:
            return candidate
    return None


def _convert_mirror_layout(mirror_dir, data_dir):
    """Convert the nnU-Net layout of the redistribution into the layout of the official release."""
    image_paths = natsorted(glob(os.path.join(mirror_dir, "imagesTr", "*_0000.nii.gz")))
    assert len(image_paths) > 0, f"Could not find any images in '{mirror_dir}'."

    for image_path in image_paths:
        case_id = os.path.basename(image_path)[:-len("_0000.nii.gz")]
        case_dir = os.path.join(data_dir, case_id)
        os.makedirs(case_dir, exist_ok=True)

        label_path = os.path.join(mirror_dir, "labelsTr", f"{case_id}.nii.gz")
        assert os.path.exists(label_path), f"Could not find the labels for case '{case_id}'."

        shutil.move(image_path, os.path.join(case_dir, "data.nii.gz"))
        shutil.move(label_path, os.path.join(case_dir, "label.nii.gz"))

    shutil.rmtree(mirror_dir)


def get_structseg_data(
    path: Union[os.PathLike, str], task: Literal["task1", "task2", "task3", "task4"], download: bool = False
) -> str:
    """Obtain the StructSeg dataset.

    Args:
        path: Filepath to a folder where the data is stored.
        task: The task of the challenge. Either 'task1', 'task2', 'task3' or 'task4'.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data of the chosen task is stored.
    """
    if task not in TASK_FOLDERS:
        raise ValueError(f"'{task}' is not a valid task. Choose one of {list(TASK_FOLDERS.keys())}.")

    data_dir = _find_task_dir(path, task)
    if data_dir is not None:
        return data_dir

    if download and task in URLS:
        os.makedirs(path, exist_ok=True)

        zip_path = os.path.join(path, os.path.basename(URLS[task]))
        util.download_source(path=zip_path, url=URLS[task], download=download, checksum=CHECKSUMS[task])
        util.unzip(zip_path=zip_path, dst=path)

        data_dir = os.path.join(path, TASK_FOLDERS[task])
        _convert_mirror_layout(os.path.join(path, MIRROR_FOLDERS[task]), data_dir)
        return data_dir

    msg = f"Could not find the StructSeg '{task}' data ('{TASK_FOLDERS[task]}') at '{path}'. "
    msg += "'torch_em' cannot download this dataset, as it requires registration for the challenge. "
    msg += "Please register at https://structseg2019.grand-challenge.org, join the challenge and follow the "
    msg += "instructions at https://structseg2019.grand-challenge.org/Download/ to obtain the archive "
    msg += f"'Task{task[-1]}_{TASK_FOLDERS[task]}.zip'. Then extract it into '{path}', such that "
    msg += f"'{os.path.join(path, TASK_FOLDERS[task], '1', 'data.nii.gz')}' exists."
    if download:
        raise NotImplementedError(msg)
    else:
        raise FileNotFoundError(msg)


def get_structseg_paths(
    path: Union[os.PathLike, str], task: Literal["task1", "task2", "task3", "task4"], download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the StructSeg data.

    Args:
        path: Filepath to a folder where the data is stored.
        task: The task of the challenge. Either 'task1', 'task2', 'task3' or 'task4'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_structseg_data(path, task, download)

    raw_paths = natsorted(glob(os.path.join(data_dir, "*", "data.nii.gz")))
    label_paths = [os.path.join(os.path.dirname(p), "label.nii.gz") for p in raw_paths]
    assert len(raw_paths) > 0 and all(os.path.exists(p) for p in label_paths)

    return raw_paths, label_paths


def get_structseg_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    task: Literal["task1", "task2", "task3", "task4"] = "task1",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the StructSeg dataset for organ-at-risk and gross target volume segmentation.

    Args:
        path: Filepath to a folder where the data is stored.
        patch_shape: The patch shape to use for training.
        task: The task of the challenge. Either 'task1', 'task2', 'task3' or 'task4'.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_structseg_paths(path, task, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key="data",
        label_paths=label_paths,
        label_key="data",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_structseg_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    task: Literal["task1", "task2", "task3", "task4"] = "task1",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the StructSeg dataloader for organ-at-risk and gross target volume segmentation.

    Args:
        path: Filepath to a folder where the data is stored.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        task: The task of the challenge. Either 'task1', 'task2', 'task3' or 'task4'.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_structseg_dataset(path, patch_shape, task, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
