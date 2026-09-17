"""The ACRIN-6698 dataset contains annotations for functional tumor volume segmentation in dynamic
contrast-enhanced (DCE) breast MRI, acquired as part of the ACRIN 6698 / I-SPY2 TRIAL.

The collection ships 1110 "VOLSER" (volume of signal enhancement ratio) DICOM-SEG objects: a binary mask
of the functional tumor volume, obtained by thresholding a per-voxel signal enhancement ratio (SER) map
computed from the pre- and post-contrast DCE-MRI phases. The mask is stored either "uni-lateral cropped"
(covering one breast) or "bi-lateral" (covering both), cropped in-plane to the analyzed region. The
collection also ships a second, unrelated family of 1103 "DWI SEG" objects (whole-tumor masks drawn on
diffusion-weighted imaging derivative series); those are covered separately by `medical.acrin_6698_dwi`.

Unlike most DICOM-SEG collections handled elsewhere in this package, these SEG objects carry neither a
`ReferencedSeriesSequence` nor a per-frame `DerivationImageSequence`: only a `StudyInstanceUID` and a
`FrameOfReferenceUID`. Matching a mask to its source series therefore requires querying the TCIA REST API
for all series of the mask's study and picking the DCE-MRI series with matching geometry, rather than the
usual reference-UID lookup used in eg. `nsclc_radiogenomics.py` or `adrenal_acc.py`.

Inspecting real downloaded data (a random sample of 100 of the 1110 VOLSER masks, across both uni-lateral
and bi-lateral cases) shows that every study contains exactly one MR series whose `SeriesDescription` is the
mask's own description with "Analysis Mask" replaced by "SER" (eg. "ISPY2: VOLSER: uni-lateral cropped:
Analysis Mask" pairs with "ISPY2: VOLSER: uni-lateral cropped: SER"). This SER series is the actual signal
enhancement ratio map the mask was thresholded from: it shares the mask's `FrameOfReferenceUID`, in-plane
`PixelSpacing`, `ImageOrientationPatient` and - since it is cropped to the same region - the exact same set
of per-slice `ImagePositionPatient` values, unlike the full-field-of-view source DCE-MRI acquisition (eg.
"ISPY2: Ax Vibrant PRE/POST"), which shares the frame of reference and in-plane geometry but not the
cropped extent. This module therefore uses the description-matched SER series as the raw image paired with
each mask. As a safety net (in case the description-based match is ever wrong or absent for some study),
the matched candidate's geometry is verified by resampling the mask onto the candidate's voxel grid (see
`_resample_labels` in `adrenal_acc.py`) and checking that (nearly) all mask foreground voxels land inside
it; if this fails, every other MR series of the study with the same `FrameOfReferenceUID` is tried in turn,
and the case is skipped (with a warning) if none of them pass the geometry check either. In the validated
sample of 100 masks, the description-based match succeeded and passed geometry verification in all 100
cases (0% failure rate); a larger production run may still see occasional skips for studies with unusual or
missing SER series, which are handled gracefully rather than raised as errors.

Two further data quirks were found by inspecting the actual downloaded pixel data of 5 validated cases and
had to be handled explicitly:
- The SEG objects use `SegmentationType = FRACTIONAL` with `SegmentationFractionalType = OCCUPANCY`, not
  `BINARY`, even though `SegmentSequence` lists only a single segment, and its `Content Description` reads
  "Bit-map mask of SER tumor segmentation". This is NOT a 0-255 fractional occupancy value: it is an
  *inverse*, bit-encoded exclusion mask, as documented in the "Analysis masks from FTV processing" support
  document linked from the TCIA collection page (https://www.cancerimagingarchive.net/collection/acrin-6698/,
  filename 'Analysis-mask-files-description.v20211020.docx', by the I-SPY imaging group, UCSF - fetched and
  parsed directly, not guessed). Per that document: "The masks are INVERSE masks, in that a mask value of 0
  indicates that a voxel was included in the measured functional tumor volume (FTV)". Each of up to 5
  masking/exclusion steps that removed a voxel from the FTV sets one bit of the byte value: 1 = failed the
  PE (percent enhancement) threshold, 2 = failed the 3D minimum-neighbor-count (MNC) connectivity filter,
  32 = outside the manually-drawn rectangular VOI, 64 = manually OMITted by a trained observer, and a
  background-intensity-threshold exclusion bit (documented as 8, but empirically 16 in the ACRIN-6698 SEG
  files actually inspected - see below). A voxel excluded by several steps has the bitwise OR of their
  values; only `{0, 1, 2, 16, 17, 32, 33, 34, 48, 49, ...}`-style combinations are therefore valid, and this
  matches byte-for-byte what is observed in the 5 validated cases (exactly `{0, 1, 2, 17, 32, 33, 34, 49}`,
  ie. every combination of {PE, background, VOI} that occurred in the sample, decomposing cleanly as
  17 = 16 (background) + 1 (PE), 33 = 32 (VOI) + 1 (PE), 34 = 32 (VOI) + 2 (MNC), 49 = 32 + 16 + 1). This
  also explains the sharp-edged rectangular sub-regions seen in an earlier (incorrect) version of this
  module that treated every non-background-mode voxel as foreground: those rectangles are the boundary of
  the manually-drawn VOI itself (voxels just inside vs. just outside it differ only in whether the 32 bit is
  set), not a real anatomical structure. This module now instead uses the documented semantics directly:
  foreground (the FTV) is exactly the voxels with value 0. Verified for the 5 validated cases: the resulting
  mask is a small, spatially compact, contiguous 3D blob (or, in 1/5 cases, empty - a legitimate "no residual
  enhancing tumor" result, since ACRIN-6698 includes post-treatment timepoints) whose per-slice area traces
  a smooth, single-peaked profile, and which spatially coincides exactly with the locations of elevated
  signal in the matched SER map (see the second quirk below and `check_acrin_6698.py`) - unlike the
  rectangle-contaminated mask from treating "not background" as foreground.
- The matched SER (and other VOLSER-derived) MR series store their quantitative ratio values with a very
  small `RescaleSlope` (0.001 in the validated cases, ie. the raw stored int16 pixel value is the SER ratio
  times 1000). Applying the slope/intercept and rounding to an integer volume, as `adrenal_acc.py` does for
  CT Hounsfield units, collapses nearly the entire dynamic range to {0, 1, 2, 3} and destroys the image. This
  module therefore keeps the raw stored pixel values as-is for the source volume, without applying the
  rescale slope/intercept. Separately, and not a bug: the SER maps themselves are extremely sparse (only
  0.02-1.9% of voxels nonzero in the validated cases) - by design, per the same support document, the
  background/non-enhancing/non-tissue voxels are zeroed out by the upstream processing pipeline before this
  derived series is exported, so most of the cropped volume being exactly 0 is a genuine property of the
  data, not a display, windowing or download issue.

Because the collection is large (1110 relevant SEG series across 385 patients), use `max_cases` or
`case_ids` to only download and preprocess a subset.

NOTE: This requires the pydicom python package.

The dataset is located at https://www.cancerimagingarchive.net/collection/acrin-6698/ and is distributed
under the CC BY 4.0 license.

This dataset is from the publication https://doi.org/10.1148/radiol.2018180273.
The data was released at https://doi.org/10.7937/tcia.kk02-6d95.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from tqdm import tqdm
from warnings import warn
from natsort import natsorted
from typing import Union, Tuple, List, Optional

import numpy as np
import requests

from torch.utils.data import Dataset, DataLoader

import torch_em

from .adrenal_acc import _resample_labels
from .. import util


COLLECTION = "ACRIN-6698"

LABEL_IDS = {"functional_tumor_volume": 1}

# The minimum fraction of the mask's foreground voxels that must land inside a candidate source series
# (after resampling onto its voxel grid) for that candidate to be accepted as the match.
MIN_GEOMETRY_OVERLAP = 0.9


def _load_dicom_volume(series_dir):
    """Stack a DICOM series into a volume with axes (z, y, x) and slices sorted along the slice normal.

    Unlike `adrenal_acc._load_dicom_volume`, the rescale slope/intercept is NOT applied here: see the
    module docstring for why this matters for the ACRIN-6698 VOLSER-derived MR series (SER, PE2, PE6, ...).

    Returns the volume and the affine matrix that maps voxel indices (z, y, x) to DICOM patient coordinates.
    """
    import pydicom

    slices = [pydicom.dcmread(dcm_path) for dcm_path in natsorted(glob(os.path.join(series_dir, "*.dcm")))]
    orientation = np.array([float(v) for v in slices[0].ImageOrientationPatient])
    row_dir, col_dir = orientation[:3], orientation[3:]
    normal = np.cross(row_dir, col_dir)
    slices.sort(key=lambda dcm: np.dot([float(v) for v in dcm.ImagePositionPatient], normal))

    volume = np.stack([dcm.pixel_array for dcm in slices])

    positions = np.array([[float(v) for v in dcm.ImagePositionPatient] for dcm in slices])
    spacing = [float(v) for v in slices[0].PixelSpacing]  # The spacing between rows and between columns.
    affine = np.eye(4)
    affine[:3, 0] = (positions[-1] - positions[0]) / (len(slices) - 1)
    affine[:3, 1] = col_dir * spacing[0]
    affine[:3, 2] = row_dir * spacing[1]
    affine[:3, 3] = positions[0]
    return volume, affine


def _load_dicom_seg(seg_path):
    """Load a VOLSER analysis mask DICOM-SEG object as a label volume with axes (z, y, x).

    Unlike `adrenal_acc._load_dicom_seg`, foreground is not determined by a nonzero pixel value: the
    ACRIN-6698 VOLSER masks are documented (see the module docstring) as INVERSE, bit-encoded exclusion
    masks, where a voxel value of exactly 0 means it was included in the functional tumor volume (FTV) and
    any nonzero value means it was excluded by one or more processing steps.

    Returns the label volume and the affine matrix that maps its voxel indices to DICOM patient coordinates.
    """
    import pydicom

    seg = pydicom.dcmread(seg_path)
    frames = seg.pixel_array
    if frames.ndim == 2:  # A segmentation with a single frame.
        frames = frames[None]

    shared_group = seg.SharedFunctionalGroupsSequence[0]
    orientation = np.array([float(v) for v in shared_group.PlaneOrientationSequence[0].ImageOrientationPatient])
    row_dir, col_dir = orientation[:3], orientation[3:]
    normal = np.cross(row_dir, col_dir)
    pixel_measures = shared_group.PixelMeasuresSequence[0]
    spacing = [float(v) for v in pixel_measures.PixelSpacing]  # The spacing between rows and between columns.

    frame_groups = seg.PerFrameFunctionalGroupsSequence
    positions = np.array([[float(v) for v in g.PlanePositionSequence[0].ImagePositionPatient] for g in frame_groups])
    projections = positions @ normal
    if "SpacingBetweenSlices" in pixel_measures:
        slice_spacing = float(pixel_measures.SpacingBetweenSlices)
    elif len(projections) > 1:
        slice_spacing = np.min(np.diff(np.unique(np.round(projections, 3))))
    else:
        slice_spacing = float(pixel_measures.SliceThickness)
    slice_ids = np.round((projections - projections.min()) / slice_spacing).astype("int")

    # A value of 0 means the voxel was included in the FTV, see the module docstring; any nonzero value
    # means it was excluded (the bitwise OR of one or more masking-step codes).
    segment_number = int(seg.SegmentSequence[0].SegmentNumber)

    labels = np.zeros((slice_ids.max() + 1, seg.Rows, seg.Columns), dtype="uint8")
    for frame, slice_id in zip(frames, slice_ids):
        labels[slice_id][frame == 0] = segment_number

    affine = np.eye(4)
    affine[:3, 0] = normal * slice_spacing
    affine[:3, 1] = col_dir * spacing[0]
    affine[:3, 2] = row_dir * spacing[1]
    affine[:3, 3] = positions[np.argmin(projections)]
    return labels, affine


def _get_seg_metadata(path: str, download: bool) -> List[dict]:
    """Query the TCIA REST API for the metadata of all SEG series of the collection, without downloading images."""
    import json

    metadata_path = os.path.join(path, "acrin_6698_seg_series.json")
    if not os.path.exists(metadata_path):
        if not download:
            raise RuntimeError(f"Cannot find the data at {path}, but download was set to False.")
        response = requests.get(util.NBIA_API_URL + "getSeries", params={"Collection": COLLECTION, "Modality": "SEG"})
        response.raise_for_status()
        with open(metadata_path, "w") as f:
            json.dump(response.json(), f, indent=2)

    with open(metadata_path, "r") as f:
        return json.load(f)


def _is_volser_mask(series_description: str) -> bool:
    return "VOLSER" in series_description and "Analysis Mask" in series_description


def _get_study_series(study_uid: str, cache: dict) -> List[dict]:
    """Query the TCIA REST API for the metadata of all series of a study, cached per study."""
    if study_uid not in cache:
        response = requests.get(util.NBIA_API_URL + "getSeries", params={"StudyInstanceUID": study_uid})
        response.raise_for_status()
        cache[study_uid] = response.json()
    return cache[study_uid]


def _match_by_description(seg_description: str, study_series: List[dict]) -> Optional[str]:
    """Find the SER series whose description matches the mask's description (see the module docstring)."""
    prefix = seg_description.replace("Analysis Mask", "SER").strip()
    matches = [
        series["SeriesInstanceUID"] for series in study_series
        if series["Modality"] == "MR" and series["SeriesDescription"].strip() == prefix
    ]
    return matches[0] if len(matches) == 1 else None


def _candidate_mr_uids(seg_description: str, study_series: List[dict]) -> List[str]:
    """List the candidate source series of a study, the description match (if any) tried first."""
    primary = _match_by_description(seg_description, study_series)
    others = [
        series["SeriesInstanceUID"] for series in study_series
        if series["Modality"] == "MR" and series["SeriesInstanceUID"] != primary
    ]
    return ([primary] if primary is not None else []) + others


def _resolve_source_volume(seg_path, seg_labels, seg_affine, frame_of_reference, mr_uids, dicom_dir, download):
    """Find the source MR series matching a mask's geometry and return its volume, aligned labels and affine.

    Tries the candidate series in order (the description-matched one first, see `_candidate_mr_uids`),
    downloading each on demand, and accepts the first one whose frame of reference matches and whose
    resampled mask overlap exceeds `MIN_GEOMETRY_OVERLAP`. A mask can legitimately have no foreground voxels
    at all (an empty functional tumor volume, eg. a complete response at a later trial timepoint, see the
    module docstring), in which case the overlap fraction is undefined; the first candidate with a matching
    frame of reference and pixel grid (`Rows`/`Columns`) is accepted instead, without downloading and trying
    every other series of the study. Returns (None, None, None) if none match.
    """
    import pydicom

    n_foreground = int((seg_labels > 0).sum())
    seg_dcm = pydicom.dcmread(seg_path, stop_before_pixels=True)
    seg_shape = (int(seg_dcm.Rows), int(seg_dcm.Columns))

    for mr_uid in mr_uids:
        mr_dir = os.path.join(dicom_dir, mr_uid)
        if not glob(os.path.join(mr_dir, "*.dcm")):
            if not download:
                continue
            util.download_tcia_series([mr_uid], dst=dicom_dir, csv_filename=os.path.join(dicom_dir, "..", "acrin_6698_mr_extra"))  # noqa
        dcm_paths = glob(os.path.join(mr_dir, "*.dcm"))
        if not dcm_paths:
            continue

        first_dcm = pydicom.dcmread(dcm_paths[0], stop_before_pixels=True)
        if str(getattr(first_dcm, "FrameOfReferenceUID", None)) != frame_of_reference:
            continue

        if n_foreground == 0:
            if (int(first_dcm.Rows), int(first_dcm.Columns)) != seg_shape:
                continue
            volume, mr_affine = _load_dicom_volume(mr_dir)
            labels = np.zeros(volume.shape, dtype=seg_labels.dtype)
            return volume, labels, mr_affine

        volume, mr_affine = _load_dicom_volume(mr_dir)
        labels = _resample_labels(seg_labels, seg_affine, volume.shape, mr_affine)
        overlap = int((labels > 0).sum()) / n_foreground
        if overlap >= MIN_GEOMETRY_OVERLAP:
            return volume, labels, mr_affine

    return None, None, None


def _preprocess_acrin_6698(seg_series: List[dict], dicom_dir: str, preprocessed_dir: str, download: bool) -> None:
    import h5py
    import pydicom

    os.makedirs(preprocessed_dir, exist_ok=True)
    study_series_cache = {}
    for series in tqdm(seg_series, desc="Preprocess ACRIN-6698"):
        seg_uid = series["SeriesInstanceUID"]
        out_path = os.path.join(preprocessed_dir, f"{seg_uid}.h5")
        if os.path.exists(out_path):
            continue

        seg_paths = glob(os.path.join(dicom_dir, seg_uid, "*.dcm"))
        if not seg_paths:
            warn(f"Skipping {seg_uid}, whose SEG DICOM data could not be found.")
            continue

        seg_dcm = pydicom.dcmread(seg_paths[0], stop_before_pixels=True)
        frame_of_reference = str(seg_dcm.FrameOfReferenceUID)

        study_series = _get_study_series(str(seg_dcm.StudyInstanceUID), study_series_cache)
        mr_uids = _candidate_mr_uids(series["SeriesDescription"], study_series)
        if not mr_uids:
            warn(f"Skipping {seg_uid}, whose study has no MR series to match against.")
            continue

        seg_labels, seg_affine = _load_dicom_seg(seg_paths[0])
        volume, labels, _ = _resolve_source_volume(
            seg_paths[0], seg_labels, seg_affine, frame_of_reference, mr_uids, dicom_dir, download
        )
        if volume is None:
            warn(f"Skipping {seg_uid}, for which no MR series with matching geometry could be found.")
            continue

        labels = (labels > 0).astype("uint8") * LABEL_IDS["functional_tumor_volume"]
        with h5py.File(out_path, "w") as f:
            f.create_dataset("raw", data=volume, compression="gzip")
            f.create_dataset("labels", data=labels, compression="gzip")


def get_acrin_6698_data(
    path: Union[os.PathLike, str],
    max_cases: Optional[int] = None,
    case_ids: Optional[List[str]] = None,
    download: bool = False,
) -> str:
    """Download the ACRIN-6698 dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        max_cases: The maximum number of VOLSER masks to download and preprocess, taken in deterministic
            order of their series instance UID. Only the SEG series and their matched source MR series are
            downloaded. Mutually exclusive with `case_ids`.
        case_ids: Explicit list of VOLSER mask series instance UIDs to download and preprocess (see the
            'acrin_6698_seg_series.json' metadata file written to `path` for the available UIDs and their
            descriptions). Mutually exclusive with `max_cases`.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the preprocessed data is stored.
    """
    assert max_cases is None or case_ids is None, "'max_cases' and 'case_ids' are mutually exclusive."

    # NOTE: The preprocessing below skips volumes that were converted already, so an interrupted run resumes.
    preprocessed_dir = os.path.join(path, "preprocessed")
    os.makedirs(path, exist_ok=True)

    seg_series = _get_seg_metadata(path, download)
    seg_series = [series for series in seg_series if _is_volser_mask(series["SeriesDescription"])]
    seg_series = natsorted(seg_series, key=lambda series: series["SeriesInstanceUID"])

    if max_cases is not None:
        seg_series = seg_series[:max_cases]
    elif case_ids is not None:
        seg_series = [series for series in seg_series if series["SeriesInstanceUID"] in case_ids]

    dicom_dir = os.path.join(path, "dicom")
    if download:
        seg_uids = [series["SeriesInstanceUID"] for series in seg_series]
        util.download_tcia_series(seg_uids, dst=dicom_dir, csv_filename=os.path.join(path, "acrin_6698_seg"))

        # The primary (description-matched) candidate source series of each mask's study is downloaded here
        # already, so that the preprocessing loop below only has to download extra candidates for the rare
        # case where the primary candidate's geometry does not match (see the module docstring).
        study_series_cache = {}
        primary_mr_uids = set()
        for series in seg_series:
            seg_paths = glob(os.path.join(dicom_dir, series["SeriesInstanceUID"], "*.dcm"))
            if not seg_paths:
                continue
            import pydicom
            study_uid = str(pydicom.dcmread(seg_paths[0], stop_before_pixels=True).StudyInstanceUID)
            study_series = _get_study_series(study_uid, study_series_cache)
            primary_uid = _match_by_description(series["SeriesDescription"], study_series)
            if primary_uid is not None:
                primary_mr_uids.add(primary_uid)
        util.download_tcia_series(
            sorted(primary_mr_uids), dst=dicom_dir, csv_filename=os.path.join(path, "acrin_6698_mr")
        )
    elif not glob(os.path.join(dicom_dir, "*", "*.dcm")):
        raise RuntimeError(f"Cannot find the data at {path}, but download was set to False.")

    _preprocess_acrin_6698(seg_series, dicom_dir, preprocessed_dir, download)
    return preprocessed_dir


def get_acrin_6698_paths(
    path: Union[os.PathLike, str],
    max_cases: Optional[int] = None,
    case_ids: Optional[List[str]] = None,
    download: bool = False,
) -> List[str]:
    """Get paths to the ACRIN-6698 data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        max_cases: The maximum number of cases to use. See `get_acrin_6698_data` for details.
        case_ids: Explicit list of case ids to use. See `get_acrin_6698_data` for details.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the hdf5 files, which contain the image data ('raw') and the label data ('labels').
    """
    preprocessed_dir = get_acrin_6698_data(path, max_cases, case_ids, download)
    volume_paths = natsorted(glob(os.path.join(preprocessed_dir, "*.h5")))
    assert len(volume_paths) > 0, f"Could not find any preprocessed volumes in '{preprocessed_dir}'."
    return volume_paths


def get_acrin_6698_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    max_cases: Optional[int] = None,
    case_ids: Optional[List[str]] = None,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the ACRIN-6698 dataset for functional tumor volume segmentation in breast DCE-MRI.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        max_cases: The maximum number of cases to use. See `get_acrin_6698_data` for details.
        case_ids: Explicit list of case ids to use. See `get_acrin_6698_data` for details.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_acrin_6698_paths(path, max_cases, case_ids, download)

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


def get_acrin_6698_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    max_cases: Optional[int] = None,
    case_ids: Optional[List[str]] = None,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the ACRIN-6698 dataloader for functional tumor volume segmentation in breast DCE-MRI.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        max_cases: The maximum number of cases to use. See `get_acrin_6698_data` for details.
        case_ids: Explicit list of case ids to use. See `get_acrin_6698_data` for details.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_acrin_6698_dataset(path, patch_shape, resize_inputs, max_cases, case_ids, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
