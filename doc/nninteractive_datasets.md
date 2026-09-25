# nnInteractive Datasets

This page lists every dataset used to train and evaluate nnInteractive (Isensee, Rokuss, Krämer et al., *nnInteractive: Redefining 3D Promptable Segmentation*, arXiv:2503.08373, https://arxiv.org/abs/2503.08373), and tracks which of them are available in `torch_em.data.datasets`.

The entries are copied verbatim from the paper's Appendix A3:
- Table A1: the training datasets. The paper caption says 120 datasets, the table itself has 114 rows.
  Note that 'CT Lymph Nodes' and 'NIH Lymph' point to the same TCIA collection.
  5% of the training images were held out for internal validation, the paper does not list which ones.
- Table A2: the held-out test datasets. These are the filtered datasets from the RadioActive benchmark
  (Ulrich et al., https://arxiv.org/abs/2411.07885) plus four additional out-of-distribution datasets.
  None of them were part of the training data.

The `torch_em` column names the module in `torch_em.data.datasets` that provides the dataset, or the proposed module name for datasets that are not integrated yet.
Status values: `available` = provided by torch-em, `partial` = torch-em has a different edition or subset, `missing` = not integrated yet.

## Table A1: Training datasets

Coverage: 109 available, 5 partial, 0 missing (out of 114 entries).

| # | Name | Images | Modality | Target | Link | Status | torch_em | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Decathlon Task 2 | 20 | MRI | Heart | http://medicaldecathlon.com | available | `medical.msd` | task 'heart' |
| 2 | Decathlon Task 3 | 131 | CT | Liver, L. Tumor | http://medicaldecathlon.com | available | `medical.msd` | task 'liver' |
| 3 | Decathlon Task 4 | 208 | MRI | Hippocampus | http://medicaldecathlon.com | available | `medical.msd` | task 'hippocampus' |
| 4 | Decathlon Task 5 | 32 | MRI | Prostate | http://medicaldecathlon.com | available | `medical.msd` | task 'prostate' |
| 5 | Decathlon Task 6 | 63 | CT | Lung Lesion | http://medicaldecathlon.com | available | `medical.msd` | task 'lung' |
| 6 | Decathlon Task 7 | 281 | CT | Pancreas, P. Tumor | http://medicaldecathlon.com | available | `medical.msd` | task 'pancreas' |
| 7 | Decathlon Task 8 | 303 | CT | Hepatic Vessel, H. Tumor | http://medicaldecathlon.com | available | `medical.msd` | task 'hepaticvessel' |
| 8 | Decathlon Task 9 | 41 | CT | Spleen | http://medicaldecathlon.com | available | `medical.msd` | task 'spleen' |
| 9 | Decathlon Task 10 | 126 | CT | Colon Tumor | http://medicaldecathlon.com | available | `medical.msd` | task 'colon' |
| 10 | ISLES2015 | 28 | MRI | Stroke Lesion | http://www.isles-challenge.org/ISLES2015 | partial | `medical.isles` | torch-em provides ISLES 2022, not the 2015 edition |
| 11 | BTCV | 30 | CT | 13 Abdominal Organs | https://www.synapse.org/Synapse:syn3193805/wiki/89480 | available | `medical.btcv` | Synapse registration required |
| 12 | LIDC | 1010 | CT | Lung Lesion | https://www.cancerimagingarchive.net/collection/lidc-idri | available | `medical.lidc_idri` | added 2026-09-14; instance labels per nodule with a selectable radiologist consensus level, validated on 30 of the 1018 CT scans since the full set is 128 GB |
| 13 | Promise12 | 50 | MRI | Prostate | https://zenodo.org/records/8026660 | available | `medical.promise12` | added 2026-09-14 |
| 14 | ACDC | 200 | MRI | RV Cavity, LV Myocardium, LV Cavity | https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html | available | `medical.acdc` |  |
| 15 | ISBILesion2015 | 42 | MRI | MS Lesion | https://iacl.ece.jhu.edu/index.php/MSChallenge | available | `medical.isbi_mslesion` | added 2026-09-14; 21 time points with masks from two raters, giving the paper's 42 volumes, downloads without the advertised registration |
| 16 | CHAOS | 60 | MRI | Liver, Kidney (L and R), Spleen | https://zenodo.org/records/3431873 | available | `medical.chaos` |  |
| 17 | BTCV 2 | 63 | CT | 9 Abdominal Organs | https://zenodo.org/records/1169361 | partial | `medical.multi_organ_abdominal_ct` | added 2026-09-14; Zenodo record holds labels for 90 cases (43 TCIA Pancreas-CT + 47 BTCV), the TCIA images download automatically, the BTCV images need Synapse registration |
| 18 | StructSeg Task1 | 50 | CT | 22 OAR Head & Neck | https://structseg2019.grand-challenge.org | available | `medical.structseg` | added 2026-09-14, validated 2026-09-14; task 'task1', all 50 volumes from an open redistribution, and the 22 organ ids are now verified empirically by tissue density, volume and left/right position, and additionally match the readme of the official release as transcribed in github.com/zhilothebest/Coronary_Calcium |
| 19 | StructSeg Task2 | 50 | CT | Nasopharynx Cancer | https://structseg2019.grand-challenge.org | available | `medical.structseg` | added 2026-09-14; challenge login required and no open copy exists, so unvalidated (task 'task2'); a second targeted search on 2026-09-15 found only the 50 unlabelled volumes in SA-Med3D-140K and confirmed the IEEE DataPort deposit holds an 83 byte url file behind a subscription; the best remaining lead is the login-gated USTC share linked from github.com/shijun18/GTV_AutoSeg, whose author also made that deposit |
| 20 | StructSeg Task3 | 50 | CT | 6 OAR Lung | https://structseg2019.grand-challenge.org | available | `medical.structseg` | added 2026-09-14; challenge login required and no open copy exists, so unvalidated (task 'task3'); a second targeted search on 2026-09-15 found only a 1.5 mm resampled copy with 4 of the 6 organs in SA-Med3D-140K, which did however let the ids 1 - 4 be verified and corrected (id 3 is the heart, not the spinal cord); the full id order is now taken from github.com/zhilothebest/Coronary_Calcium, which transcribes the readme of the official release |
| 21 | StructSeg Task4 | 50 | CT | Lung Cancer | https://structseg2019.grand-challenge.org | available | `medical.structseg` | added 2026-09-14; challenge login required and no open copy exists, so unvalidated (task 'task4'); a second targeted search on 2026-09-15 found nothing at all of this task, and the IEEE DataPort deposit that covers it needs a subscription and holds only a url file; the best remaining lead is the login-gated USTC share linked from github.com/shijun18/GTV_AutoSeg, whose author also made that deposit |
| 22 | SegTHOR | 40 | CT | Heart, Aorta, Trachea, Esophagus | https://competitions.codalab.org/competitions/21145 | available | `medical.segthor` | added 2026-09-14; all 40 volumes from a public Zenodo mirror, label meanings verified anatomically |
| 23 | NIH-Pan | 82 | CT | Pancreas | https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT | available | `medical.nih_pancreas` | added 2026-09-14; requires pydicom |
| 24 | VerSe2020 | 113 | CT | 28 Vertebrae | https://github.com/anjany/verse | available | `medical.verse` |  |
| 25 | M&Ms | 300 | MRI | Left & Right Ventricle, Myocardium | https://www.ub.edu/mnms | available | `medical.mnms` | added 2026-09-14; 320 annotated studies giving 640 end-diastole and end-systole volumes, the paper's 300 is the training split, downloaded from a HuggingFace mirror since the official site is offline. NOTE the label order is the reverse of ACDC: 1 left ventricle cavity, 2 myocardium, 3 right ventricle |
| 26 | ProstateX | 140 | MRI | Prostate Lesion | https://www.aapm.org/GrandChallenge/PROSTATEx-2 | available | `medical.prostatex` | added 2026-09-14; 204 patients, lesion masks from the Cuocolo PROSTATEx_masks repository plus prostate zones |
| 27 | RibSeg | 370 | CT | Ribs | https://github.com/M3DV/RibSeg?tab=readme-ov-file | available | `medical.ribseg` | added 2026-09-14; RibSeg v2 labels from Google Drive paired with the public RibFrac images, ids 1 to 24 are one per rib |
| 28 | BrainMetShare | 84 | MRI | Brain Metastases | https://aimi.stanford.edu/brainmetshare | available | `medical.brainmetshare` | added and validated 2026-09-14; 105 labelled studies from a Kaggle mirror, which needs a Kaggle API token. The underlying data stays subject to Stanford's research use agreement |
| 29 | CrossModa22 | 168 | MRI | Vestibular Schwannoma, Cochlea | https://crossmoda2022.grand-challenge.org | available | `medical.crossmoda` | added 2026-09-14; labelled ceT1 source domain only |
| 30 | Atlas22 | 524 | MRI | Stroke Lesion | https://atlas.grand-challenge.org | available | `medical.atlas_stroke` | added 2026-09-14; all 655 public training volumes from a HuggingFace mirror since the OpenNeuro copy was deleted. NOTE 655 is the real count, the paper's 524 is not an ATLAS figure |
| 31 | KiTs23 | 489 | CT | Kidneys, K. Tumor, Cysts | https://kits-challenge.org/kits23 | available | `medical.kits` |  |
| 32 | AutoPet2 | 1014 | PET,CT | Lesions | https://autopet-ii.grand-challenge.org | available | `medical.autopet` |  |
| 33 | AMOS | 360 | CT,MRI | 15 Abdominal Organs | https://amos22.grand-challenge.org | available | `medical.amos` |  |
| 34 | BraTS24 | 1251 | MRI | Glioblastoma | https://www.synapse.org/Synapse:syn51156910/wiki/621282 | available | `medical.brats` | added 2026-09-14; the 1251-case adult glioma release the paper links to, from a public mirror since Synapse is gated. Labels are 1 necrotic core, 2 edema, 3 enhancing, with no id 4 as in BraTS 2021. A region argument builds whole tumor and tumor core as unions |
| 35 | AbdomenAtlas1.1Mini | 5195 | CT | 8 Abdominal Organs | https://huggingface.co/datasets/AbdomenAtlas/_AbdomenAtlas1.1Mini | available | `medical.abdomen_atlas` | added 2026-09-14, validated 2026-09-15; all 5195 volumes with 25 classes, downloaded after accepting the HuggingFace terms. Needs a token via the 'token' argument or the HF_TOKEN variable |
| 36 | TotalSegmentatorV2 | 1180 | CT | 117 Classes of Whole Body | https://github.com/wasserth/TotalSegmentator | available | `medical.totalsegmentator` | added 2026-09-14; dataset v2.0.1 with 1228 CTs |
| 37 | Hecktor2022 | 524 | PET,CT | Head and Neck Tumor | https://hecktor.grand-challenge.org | available | `medical.hecktor` | added 2026-09-14; the organizers have the data offline and every open mirror is a different edition (2025, 2026), so unvalidated |
| 38 | FLARE | 50 | CT | 13 Abdominal Organs | https://flare22.grand-challenge.org | available | `medical.flare` | added 2026-09-14; all 50 labelled FLARE22 volumes from the organizers' Zenodo record, FLARE23 and FLARE24 would be separate modules |
| 39 | SegA | 56 | CT | Aorta | https://multicenteraorta.grand-challenge.org/data | available | `medical.sega` |  |
| 40 | WORD | 120 | CT | 16 Abdominal Organs | https://github.com/HiLab-git/WORD | available | `medical.word` | added 2026-09-14, validated 2026-09-14; all 150 volumes with the official 100/20/30 split, the Google Drive download is quota limited so it may need a retry |
| 41 | AbdomenCT1K | 996 | CT | Liver, Kidney, Spleen, Pancreas | https://github.com/JunMa11/AbdomenCT-1K | available | `medical.abdomenct_1k` | added 2026-09-14; 1000 of 1112 volumes have public labels |
| 42 | DAP-ATLAS | 533 | CT | 142 Classes of Whole Body | https://github.com/alexanderjaus/AtlasDataset | available | `medical.dap_atlas` | added 2026-09-14, fully validated 2026-09-15; all 533 volumes with 144 classes, labels from Google Drive and images reused from medical.autopet |
| 43 | CTORG | 140 | CT | Lung, Brain, Bones, Liver, Kidney, Bladder | https://www.cancerimagingarchive.net/collection/ct-org | available | `medical.ct_org` | added 2026-09-14; downloaded from the HuggingFace mirror MedOtter/ct-org since TCIA only offers Aspera |
| 44 | TopCow | 200 | CT,MRI | Vessel Components of CoW | https://topcow23.grand-challenge.org | available | `medical.topcow` | added 2026-09-14; the 2024 release has 250 annotated scans, 125 CT and 125 MR, more than the 200 of the paper |
| 45 | AortaSeg24 | 50 | CT | Aorta | https://aortaseg24.grand-challenge.org | available | `medical.aortaseg24` | added 2026-09-14; data agreement required so unvalidated, the 23 aortic segment ids come from the official evaluation script |
| 46 | Duke Liver | 310 | MRI | Liver | https://zenodo.org/records/7774566 | available | `medical.duke_liver` |  |
| 47 | Aero Path | 27 | CT | Lungs, Airways | https://github.com/raidionics/AeroPath | available | `medical.aeropath` | added 2026-09-14 |
| 48 | AxonEM | 18 | El. Microscopy | Axon Instances | https://axonem.grand-challenge.org | available | `electron_microscopy.axonem` |  |
| 49 | MitoEM | 4 | El. Microscopy | Mitochondria Instances | https://mitoem.grand-challenge.org | available | `electron_microscopy.mitoem` |  |
| 50 | NucMM | 62 | El. Microscopy | Neuronal Nuclei | https://nucmm.grand-challenge.org | available | `electron_microscopy.nuc_mm` |  |
| 51 | LungVis1.0 | 22 | Fl. Microscopy | Airway | https://zenodo.org/records/7413818 | available | `light_microscopy.lungvis` | added 2026-09-14; Zenodo has 78 lungs, 20 with human-verified airway labels (default), 58 with AI labels |
| 52 | BBBC024 HL60 Cell line | 240 | Fl. Microscopy | Cell Nuclei | https://bbbc.broadinstitute.org/BBBC024 | available | `light_microscopy.bbbc024` | added 2026-09-14 |
| 53 | BBBC027 Colon Tissue | 60 | Fl. Microscopy | Colon Tissue | https://bbbc.broadinstitute.org/BBBC027 | available | `light_microscopy.bbbc027` | added 2026-09-14; ground truth is a binary foreground mask, not instances |
| 54 | BBBC032 MouseEmbryoBlastocyst | 1 | Fl. Microscopy | Blastocyst Cells | https://bbbc.broadinstitute.org/BBBC032 | available | `light_microscopy.bbbc032` | added 2026-09-14; sparse instance annotations |
| 55 | BBBC033 MouseTrophoblast | 1 | Microscopy | Trophoblast | https://bbbc.broadinstitute.org/BBBC033 | available | `light_microscopy.bbbc033` | added 2026-09-14 |
| 56 | BBBC034 PluripStemCells | 1 | Microscopy | Stem Cells | https://bbbc.broadinstitute.org/BBBC034 | available | `light_microscopy.bbbc034` |  |
| 57 | BBBC046 FiloData3D | 5400 | Fl. Microscopy | Lung Cancer Cells | https://bbbc.broadinstitute.org/BBBC046 | available | `light_microscopy.bbbc046` | added 2026-09-14; 5400 volumes in 9 archives (36 GB) |
| 58 | BBBC050 MouseEmbryoNuclei | 165 | Fl. Microscopy | Mouse Embryonic Cells | https://bbbc.broadinstitute.org/BBBC050 | available | `light_microscopy.bbbc050` | added 2026-09-14 |
| 59 | CAMUS | 1000 | US | Endocardium, Epicardium, Atrium | https://www.creatis.insa-lyon.fr/Challenge/camus/index.html | available | `medical.camus` |  |
| 60 | CETUS | 90 | US | LV Lumen | https://www.creatis.insa-lyon.fr/Challenge/CETUS/databases.html | available | `medical.cetus` | added 2026-09-14; all 90 volumes, public on the same CREATIS server as CAMUS despite the dead registration page |
| 61 | EPFL Mito | 1 | El. Microscopy | Mitochondria | https://www.epfl.ch/labs/cvlab/data/data-em | available | `electron_microscopy.lucchi` |  |
| 62 | FETA | 120 | MRI | Brain Regions | https://fetachallenge.github.io/pages/Data_description | available | `medical.feta24` | Synapse registration required |
| 63 | Drosophila | 1 | El. Microscopy | Mitochondria, Synapses | https://github.com/unidesigner/groundtruth-drosophila-vnc | available | `electron_microscopy.vnc` |  |
| 64 | Leg3DUS | 44 | US | Lower-Limb Leg | https://www.cs.cit.tum.de/camp/publications/leg-3d-us-dataset | available | `medical.leg_3d_us` |  |
| 65 | LGGMRISeg | 110 | MRI | Tumor | https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation/data | available | `medical.lgg_mri` |  |
| 66 | M-CRIB | 10 | MRI | Neonatal Brain Atlas | https://osf.io/4vthr | available | `medical.mcrib` | added 2026-09-14 |
| 67 | ParticleSeg3D | 54 | MicroCT | Mineral Samples | https://syncandshare.desy.de/index.php/s/wjiDQ49KangiPj5 | available | `medical.particleseg3d` | added 2026-09-14; 54 annotated patches with instance labels, downloaded from the DESY Nextcloud share |
| 68 | RESECT | 69 | US | Cerebral Tumor | https://osf.io/jv8bk | partial | `medical.resect` | added 2026-09-14; RESECT-SEG has tumor labels only for the 23 pre-resection US volumes, the during/after volumes carry resection-cavity labels |
| 69 | CAP | 1637 | MRI | Left Ventricle | https://www.cardiacatlas.org/lv-segmentation-challenge | available | `medical.cap_lv` | added 2026-09-14; no open copy, the cohort is unavailable while its sharing agreement is renewed. The file naming is now confirmed from the challenge FAQ and slices are ordered by DICOM position. Unvalidated |
| 70 | AtriaSeg2018 | 100 | MRI | Left Atrium | https://www.cardiacatlas.org/atriaseg2018-challenge/atria-seg-data | available | `medical.atriaseg` | added 2026-09-14; 154 volumes since the test split also ships labels, left atrium cavity and wall |
| 71 | NIS3D | 6 | Fl. Microscopy | Cell Nuclei | https://zenodo.org/records/11456029 | available | `light_microscopy.nis3d` |  |
| 72 | SegThy 1 | 14 | MRI | Thyroid, Carotid, Jugular Vein | https://www.cs.cit.tum.de/camp/publications/segthy-dataset | available | `medical.segthy` | source 'MRI' |
| 73 | SegThy 2 | 32 | US | Thyroid, Carotid, Jugular Vein | https://www.cs.cit.tum.de/camp/publications/segthy-dataset | available | `medical.segthy` | source 'US' |
| 74 | Fluo C3DH A549 | 90 | Fl. Microscopy | Cell | https://celltrackingchallenge.net/3d-datasets | available | `light_microscopy.ctc` | added 2026-09-14; 3D CTC datasets supported, name 'Fluo-C3DH-A549' |
| 75 | Fluo N3DH | 230 | Fl. Microscopy | Cell, Border | https://celltrackingchallenge.net/3d-datasets | available | `light_microscopy.ctc` | added 2026-09-14; 'Fluo-N3DH-SIM+' has 230 full-volume annotations, 'Fluo-N3DH-CE' and 'Fluo-N3DH-CHO' only have slice annotations (use annotation_type='ST' for CE volumes) |
| 76 | Spine-Mets | 55 | CT | Vertebra | https://www.cancerimagingarchive.net/collection/spine-mets-ct-seg | available | `medical.spine_mets` | added 2026-09-14; requires pydicom |
| 77 | WMHSegChallenge | 60 | MRI | White Matter Hyperintensities | https://dataverse.nl/dataset.xhtml?persistentId=doi:10.34894/AECRSD | available | `medical.wmh` | added 2026-09-14 |
| 78 | NCI-ISBI | 59 | MRI | Prostate | https://www.cancerimagingarchive.net/analysis-result/isbi-mr-prostate-2013 | available | `medical.nci_isbi_prostate` | added 2026-09-14; all 80 volumes with the official three-way split, 13 label files needed resampling onto the image grid |
| 79 | OASIS | 436 | MRI | Brain Regions | https://sites.wustl.edu/oasisbrains/home/oasis-1 | available | `medical.oasis` |  |
| 80 | MediaLymph | 15 | CT | Mediastinal Lymph Nodes | https://github.com/dbouget/ct_mediastinal_structures_segmentation | available | `medical.mediastinal_ct` | added 2026-09-14; task 'lymph_nodes' |
| 81 | MediaStruct | 15 | CT | Mediastinal Structures | https://github.com/dbouget/ct_mediastinal_structures_segmentation | available | `medical.mediastinal_ct` | added 2026-09-14; task 'structures' |
| 82 | CT Lymph Nodes | 175 | CT | Lymph Nodes | https://www.cancerimagingarchive.net/collection/ct-lymph-nodes | available | `medical.ct_lymph_nodes` | added 2026-09-14; all 176 volumes with per-node instance ids |
| 83 | MAMA MIA | 1506 | MRI | Breast Lesions | https://www.synapse.org/Synapse:syn60868042/wiki/628716 | available | `medical.mama_mia` | added 2026-09-14; all 1506 volumes from an open mirror since Synapse is gated, expert segmentations and the first post-contrast phase only |
| 84 | ATM2022 | 300 | CT | Airway Tree | https://atm22.grand-challenge.org | available | `medical.atm22` | added 2026-09-14; 280 of 300 cases, the organizers withhold the 20 EXACT'09 images |
| 85 | Pediatric CT SEG | 353 | CT | Organs | https://doi.org/10.7937/TCIA.X0H0-1706 | available | `medical.pediatric_ct_seg` | added 2026-09-14; all 359 volumes converted, 29 organ classes from RTSTRUCT contours |
| 86 | Atlas Bourgogne | 60 | MRI | Liver, Tumor | https://atlas-challenge.u-bourgogne.fr | available | `medical.atlas_liver` | added 2026-09-14, validated 2026-09-15; all 60 volumes, obtained manually after signing up since the download needs a session. Label ids confirmed against the official dataset.json, and the liver and tumor labels are disjoint |
| 87 | CC Tumor Heterogeneity | 63 | MRI | Cervix, Tumor | https://www.cancerimagingarchive.net/collection/cc-tumor-heterogeneity | available | `medical.cc_tumor_heterogeneity` | added 2026-09-14; 68 volumes, tumor and cervix are distinguished by the contour display colour |
| 88 | CURVAS | 60 | CT | Pancreas, Kidney, Liver | https://curvas.grand-challenge.org/curvas-dataset | available | `medical.curvas` |  |
| 89 | Emidec | 100 | MRI | Heart Structures | https://emidec.com | available | `medical.emidec` | added 2026-09-14; all 100 annotated volumes, downloads without login despite the registration page |
| 90 | HVSMR-2.0 | 60 | MRI | Heart, Vessel | https://segchd.csail.mit.edu | available | `medical.hvsmr` | added 2026-09-14; all 60 volumes. NOTE version 2.0 labels four chambers and four great vessels, not the blood pool and myocardium of HVSMR 2016 |
| 91 | Kipa22 | 70 | CT | Kidney, Vessel, Tumor | https://kipa22.grand-challenge.org | available | `medical.kipa` | added 2026-09-14; all 70 volumes from a public mirror, label volumes verified against the official statistics |
| 92 | MrBrains18 | 30 | MRI | Brain Structures | https://mrbrains18.isi.uu.nl/index.html | available | `medical.mrbrains18` | added 2026-09-14; all 30 volumes since the test split also ships labels, from DataverseNL as the challenge site is dead |
| 93 | OrCaScore | 32 | CT | Calcifications | https://orcascore.grand-challenge.org | available | `medical.orcascore` | added 2026-09-14; never openly distributed, the archived download page confirms the layout but holds no data. Unvalidated |
| 94 | Parse22 | 100 | CT | Pulmonary Artery | https://parse2022.grand-challenge.org/Parse2022 | available | `medical.parse22` | added 2026-09-14; all 100 training volumes from the organizer-published Drive link |
| 95 | PDDCA | 47 | CT | Head and Neck Structures | https://www.imagenglab.com/newsite/pddca | available | `medical.pddca` | added 2026-09-14 |
| 96 | ProstateEdgeCases | 131 | CT | Bladder, Prostate, Rectum | https://www.cancerimagingarchive.net/collection/prostate-anatomical-edge-cases | available | `medical.prostate_edge_cases` | added 2026-09-14; all 131 volumes, femoral heads kept as ids 4 and 5 |
| 97 | SKI10 | 100 | MRI | Cartilage, Bone | https://ski10.grand-challenge.org | available | `medical.ski10` | added 2026-09-14; all 100 training volumes from an open mirror since the challenge is closed, label ids confirmed three independent ways |
| 98 | Soft Tissue Sarcoma | 102 | MRI | Edema, Tumor | https://www.cancerimagingarchive.net/collection/soft-tissue-sarcoma | available | `medical.soft_tissue_sarcoma` | added 2026-09-14; 51 volumes each for T1 and T2FS, matching the paper's 102 MRIs, plus CT and PET |
| 99 | Spider | 447 | MRI | Lumbar Spine | https://zenodo.org/records/10159290 | available | `medical.spider` |  |
| 100 | VALDO Task 2 | 72 | MRI | Cerebral Microbleed | https://valdo.grand-challenge.org/Task2 | available | `medical.valdo` | added 2026-09-14; all 72 volumes, labels are binary rather than per-microbleed instances |
| 101 | ToothFairy 2 | 480 | CT | Dental Structures | https://ditto.ing.unimore.it/toothfairy2 | available | `medical.toothfairy` | version 'v2', registration required |
| 102 | UPENN-GBM | 147 | MRI | Edema, Tumor | https://www.cancerimagingarchive.net/collection/upenn-gbm | available | `medical.upenn_gbm` | added 2026-09-14, both variants validated 2026-09-15; 147 manual and 611 automated segmentations from the HuggingFace mirror MedOtter/UPENN-GBM since TCIA only offers Aspera. Labels use the older BraTS convention with ids 1, 2 and 4, unlike medical.brats which uses 1, 2 and 3 |
| 103 | ReMIND | 213 | MRI | Brain Resection | https://www.cancerimagingarchive.net/collection/remind | available | `medical.remind` | added 2026-09-14; 221 annotated MRI series (the paper counts 213), only the annotated series are downloaded |
| 104 | Prostate158 | 188 | MRI | Gland, Tumor | https://zenodo.org/records/6481141 | available | `medical.prostate158` | added 2026-09-14 |
| 105 | TotalSegmentator MRI | 298 | MRI | Organs | https://zenodo.org/records/11367005 | partial | `medical.totalsegmentator_mri` | added 2026-09-14; torch-em uses dataset v3.0.0 (1296 MRIs, 50 classes), the paper used the 298-image v1 |
| 106 | Instance2022 | 100 | CT | Intracranial Hemorrhage | https://instance.grand-challenge.org | available | `medical.instance22` | added 2026-09-14, validated 2026-09-15; all 100 volumes, obtained manually after joining the challenge since the page refuses anonymous access and the licence forbids redistribution |
| 107 | LAPD Mouse | 34 | Fl. Microscopy | Airway | https://cebs-ext.niehs.nih.gov/cahs/report/lapd/web-download-links | available | `light_microscopy.lapd_mouse` | added 2026-09-14; default is the 4x subsampled resolution |
| 108 | Deep Lesion | 1093 | CT | Multiple Types of Lesions | https://nihcc.app.box.com/v/DeepLesion | partial | `medical.deeplesion` | added 2026-09-14; uses the public ULS23 DeepLesion3D derivative with 743 lesion sub-volumes, the 1093 full-volume 3D masks of the paper are not public |
| 109 | COVID-19 CT Lung | 10 | CT | COVID -19 | https://zenodo.org/records/3757476 | available | `medical.covid19_seg` |  |
| 110 | LNDb | 229 | CT | Lymph Nodes | https://lndb.grand-challenge.org | available | `medical.lndb` | added 2026-09-14; 236 volumes with per-nodule instance ids and a selectable radiologist consensus level. The paper's 294 includes 58 test scans that have centroids but no segmentations |
| 111 | NIH Lymph | 176 | CT | Lymph Nodes | https://www.cancerimagingarchive.net/collection/ct-lymph-nodes | available | `medical.ct_lymph_nodes` | added 2026-09-14; same TCIA collection as 'CT Lymph Nodes', listed twice in the paper |
| 112 | NSCLC Pleural Effusion | 78 | CT | Pleural Effusion | https://www.cancerimagingarchive.net/analysis-result/plethora | available | `medical.plethora` |  |
| 113 | NSCLC Radiomics | 503 | CT | Lung Lesions | https://www.cancerimagingarchive.net/collection/nsclc-radiomics | available | `medical.nsclc_radiomics` | added 2026-09-14; all 422 volumes, the paper's 503 appears to count lesions rather than volumes |
| 114 | COVID-19-20 | 199 | CT | COVID-19 | https://covid-segmentation.grand-challenge.org/COVID-19-20 | available | `medical.covid19_20` | added 2026-09-14; all 199 volumes, kept separate from the 20-case medical.covid19_seg which the paper lists as 'COVID-19 CT Lung' |

## Table A2: Held-out test datasets

Coverage: 11 available, 3 partial, 0 missing (out of 14 entries).

| # | Group | Name | Modality | Target | Images | Status | torch_em | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | RadioActive Benchmark Datasets [135] | MS Lesion | MRI (T2 Flair) | MS Lesions | 60 | available | `medical.mendeley_ms` | added 2026-09-14; identified as the Muslim et al. Mendeley multiple sclerosis dataset, 60 patients with T1, T2 and FLAIR each separately labelled |
| 2 | RadioActive Benchmark Datasets [135] | HanSeg | MRI (T1) | 30 Organs at Risk | 42 | partial | `medical.han_seg` | torch-em only provides the CT scans, the paper evaluates on the MR T1 scans |
| 3 | RadioActive Benchmark Datasets [135] | HNTSRMFG | MRI (T2) | Oropharyngeal Cancer and Metastatic Lymph Nodes | 135 | available | `medical.hntsmrg` | added 2026-09-14; HNTS-MRG 2024 training set, pre- and mid-RT |
| 4 | RadioActive Benchmark Datasets [135] | RiderLung | CT | Lung Lesions | 58 | available | `medical.rider_lung` | added 2026-09-14; 59 CT volumes with manual and automatic tumor contours |
| 5 | RadioActive Benchmark Datasets [135] | LNQ | CT | Mediastinal Lymph Nodes | 513 | available | `medical.lnq` | added 2026-09-14; all 513 volumes from the public TCIA release, the challenge nrrd distribution needs a login |
| 6 | RadioActive Benchmark Datasets [135] | LiverMets | CT | Liver Metastases | 171 | available | `medical.colorectal_liver_mets` | added 2026-09-14; all 197 volumes, liver, hepatic vein, portal vein and tumor |
| 7 | RadioActive Benchmark Datasets [135] | Adrenal ACC | CT | Adrenal Tumors | 53 | available | `medical.adrenal_acc` | added 2026-09-14; all 53 volumes |
| 8 | RadioActive Benchmark Datasets [135] | HCC Tace | CT | Liver and Liver Tumors | 65 | available | `medical.hcc_tace` | added 2026-09-14; all 105 volumes, only the segmented CT phase is downloaded |
| 9 | RadioActive Benchmark Datasets [135] | Pengwin | CT | Bone Fragments | 100 | available | `medical.pengwin` | modality 'CT' |
| 10 | RadioActive Benchmark Datasets [135] | SegRap | CT | 45 Organs at Risk | 30 | available | `medical.segrap` | added and validated 2026-09-14; all 120 cases from an open mirror of the organizers' own release. NOTE the official Task001 labels are a single volume with 54 disjoint sub-part ids, not 45 merged organ masks, cross-checked against the challenge's own postprocessing code |
| 11 | Additional OOD Datasets | MouseTumor | MicroCT | Subcutaneous Tumors in Mice | 452 | available | `medical.mice_tumseg` |  |
| 12 | Additional OOD Datasets | InsectAnatomy | MicroCT | Insect Brain | 84 | partial | `medical.insect_anatomy` | added 2026-09-14, data obtained 2026-09-15; the public Dryad deposit holds 2D cross-sections of ant head scans, not the 3D microCT volumes the paper used. 35422 train and 7366 test pairs. NOTE the test masks are unreliable, most of them mark the background instead of the brain, so only the train split is recommended |
| 13 | Additional OOD Datasets | ACRIN H&N | PET | Head and Neck Tumors | 67 | partial | `medical.acrin_hnscc` | added 2026-09-14; the tumor annotations are public and download automatically, but the images need NIH controlled data access so the data function raises with manual steps |
| 14 | Additional OOD Datasets | Stanford Knee | MRI | Patellar, Femoral, Tibial Cartilages and Meniscus | 155 | available | `medical.skm_tea` | added 2026-09-14; only the first author's official 3-scan sample is openly downloadable and the module was validated on it, the full 155-scan release still needs a Stanford AIMI account |

## Paper references for the bracketed citation numbers

The paper cites each dataset with reference numbers. They are kept here for cross-checking against the paper's bibliography:

| Name | Reference numbers in the paper |
| --- | --- |
| Decathlon Task 2 | 5, 125 |
| Decathlon Task 3 | 5, 125 |
| Decathlon Task 4 | 5, 125 |
| Decathlon Task 5 | 5, 125 |
| Decathlon Task 6 | 5, 125 |
| Decathlon Task 7 | 5, 125 |
| Decathlon Task 8 | 5, 125 |
| Decathlon Task 9 | 5, 125 |
| Decathlon Task 10 | 5, 125 |
| ISLES2015 | 91 |
| BTCV | 67 |
| LIDC | 7 |
| Promise12 | 77 |
| ACDC | 15 |
| ISBILesion2015 | 23 |
| CHAOS | 58 |
| BTCV 2 | 37 |
| StructSeg Task1 | 69 |
| StructSeg Task2 | 69 |
| StructSeg Task3 | 69 |
| StructSeg Task4 | 69 |
| SegTHOR | 66 |
| NIH-Pan | 25, 119 |
| VerSe2020 | 123, 81, 74 |
| M&Ms | 22, 93 |
| ProstateX | 78 |
| RibSeg | 147 |
| BrainMetShare | 40 |
| CrossModa22 | 124 |
| Atlas22 | 75 |
| KiTs23 | 43 |
| AutoPet2 | 34 |
| AMOS | 50 |
| BraTS24 | 57, 9, 96, 8 |
| AbdomenAtlas1.1Mini | 71, 108 |
| TotalSegmentatorV2 | 141 |
| Hecktor2022 | 4 |
| FLARE | 87 |
| SegA | 110, 51, 104 |
| WORD | 84, 73 |
| AbdomenCT1K | 86 |
| DAP-ATLAS | 48 |
| CTORG | 114 |
| TopCow | 148 |
| AortaSeg24 | 45 |
| Duke Liver | 90 |
| Aero Path | 128 |
| AxonEM | 142 |
| MitoEM | 142 |
| NucMM | 76 |
| LungVis1.0 | 149 |
| BBBC024 HL60 Cell line | 131 |
| BBBC027 Colon Tissue | 132 |
| BBBC032 MouseEmbryoBlastocyst | 80 |
| BBBC033 MouseTrophoblast | 80 |
| BBBC034 PluripStemCells | 80 |
| BBBC046 FiloData3D | 80 |
| BBBC050 MouseEmbryoNuclei | 80 |
| CAMUS | 68 |
| CETUS | 14 |
| EPFL Mito | 82 |
| FETA | 102 |
| Drosophila | 36 |
| Leg3DUS | 31 |
| LGGMRISeg | 21 |
| M-CRIB | 3 |
| ParticleSeg3D | 39 |
| RESECT | 12 |
| CAP | 55 |
| AtriaSeg2018 | 145 |
| NIS3D | 153 |
| SegThy 1 | 62 |
| SegThy 2 | 62 |
| Fluo C3DH A549 | 94 |
| Fluo N3DH | 94 |
| Spine-Mets | 105 |
| WMHSegChallenge | 64 |
| NCI-ISBI | 16 |
| OASIS | 92 |
| MediaLymph | 19 |
| MediaStruct | 18 |
| CT Lymph Nodes | 118 |
| MAMA MIA | 33 |
| ATM2022 | 151 |
| Pediatric CT SEG | 52 |
| Atlas Bourgogne | 109 |
| CC Tumor Heterogeneity | 95 |
| CURVAS | 113 |
| Emidec | 65 |
| HVSMR-2.0 | 101 |
| Kipa22 | 41 |
| MrBrains18 | 63 |
| OrCaScore | 143 |
| Parse22 | 83 |
| PDDCA | 111 |
| ProstateEdgeCases | 56 |
| SKI10 | 138 |
| Soft Tissue Sarcoma | 136 |
| Spider | 137 |
| VALDO Task 2 | 129 |
| ToothFairy 2 | 17 |
| UPENN-GBM | 10 |
| ReMIND | 54 |
| Prostate158 | 130 |
| TotalSegmentator MRI | 26 |
| Instance2022 | 72 |
| LAPD Mouse | 13 |
| Deep Lesion | 146 |
| COVID-19 CT Lung | 53 |
| LNDb | 103 |
| NIH Lymph | 117 |
| NSCLC Pleural Effusion | 61 |
| NSCLC Radiomics | 2 |
| COVID-19-20 | 120 |
| MS Lesion | 99 |
| HanSeg | 106 |
| HNTSRMFG | 139 |
| RiderLung | 152 |
| LNQ | 29 |
| LiverMets | 126 |
| Adrenal ACC | 98 |
| HCC Tace | 97 |
| Pengwin | 79 |
| SegRap | 85 |
| MouseTumor | 49 |
| InsectAnatomy | 133 |
| ACRIN H&N | 59 |
| Stanford Knee | 28 |
