import os
import requests
from glob import glob
from natsort import natsorted

import numpy as np
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

        df = nbia.downloadSeries(trg_path, input_type="manifest", number=3, format="df", path=ROOT)

    breakpoint()

    all_patient_dirs = glob(os.path.join(ROOT, "*"))
    for patient_dir in all_patient_dirs:
        if not os.path.split(patient_dir)[-1].startswith("1.3"):
            continue

        all_dicom_files = natsorted(glob(os.path.join(patient_dir, "*.dcm")))
        samples = []
        for dcm_fpath in all_dicom_files:
            file = dicom.dcmread(dcm_fpath)
            img = file.pixel_array
            samples.append(img)

        samples = np.stack(samples)

        import napari

        v = napari.Viewer()
        v.add_image(samples)
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
