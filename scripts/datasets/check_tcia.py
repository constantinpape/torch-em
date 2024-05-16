import os
import requests
from glob import glob
from natsort import natsorted

import numpy as np
import pandas as pd
import nibabel as nib
import pydicom as dicom

from tcia_utils import nbia


ROOT = "/media/anwai/ANWAI/data/tmp/"

TCIA_URL = "https://wiki.cancerimagingarchive.net/download/attachments/68551327/NSCLC-Radiomics-OriginalCTs.tcia"


def check_tcia(download):
    trg_path = os.path.join(ROOT, os.path.split(TCIA_URL)[-1])
    if download:
        # output = nbia.getSeries(collection="LIDC-IDRI")
        # nbia.downloadSeries(output, number=3, path=ROOT)

        manifest = requests.get(TCIA_URL)
        with open(trg_path, 'wb') as f:
            f.write(manifest.content)

        nbia.downloadSeries(
            series_data=trg_path, input_type="manifest", number=3, path=ROOT, csv_filename="save"
        )

    df = pd.read_csv("save.csv")

    all_patient_dirs = glob(os.path.join(ROOT, "*"))
    for patient_dir in all_patient_dirs:
        patient_id = os.path.split(patient_dir)[-1]
        if not patient_id.startswith("1.3"):
            continue

        breakpoint()

        subject_id = pd.Series.to_string(df.loc[df["Series UID"] == patient_id]["Subject ID"])[-9:]
        seg_path = glob(os.path.join(ROOT, "Thoracic_Cavities", subject_id, "*.nii.gz"))[0]
        gt = nib.load(seg_path)
        gt = gt.get_fdata()
        gt = gt.transpose(2, 1, 0)
        gt = np.flip(gt, axis=1)

        all_dicom_files = natsorted(glob(os.path.join(patient_dir, "*.dcm")))
        samples = []
        for dcm_fpath in all_dicom_files:
            file = dicom.dcmread(dcm_fpath)
            img = file.pixel_array
            samples.append(img)

        samples = np.stack(samples[::-1])

        import napari

        v = napari.Viewer()
        v.add_image(samples)
        v.add_labels(gt.astype("uint64"))
        napari.run()


def _test_me():
    data = nbia.getSeries(collection="Soft-tissue-Sarcoma")
    print(data)

    nbia.downloadSeries(data, number=3)

    seriesUid = "1.3.6.1.4.1.14519.5.2.1.5168.1900.104193299251798317056218297018"
    nbia.viewSeries(seriesUid)


if __name__ == "__main__":
    # _test_me()
    check_tcia(download=False)
