"""The STHELAR dataset contains annotations for nucleus instance segmentation and cell type classification
in H&E stained histopathology images of 13 human tissue types (breast, cervix, colon, heart, kidney, liver,
lung, lymph node, ovary, pancreas, prostate, skin and tonsil).

The dataset links Xenium spatial transcriptomics with H&E whole slide images. The nuclei masks and the cell types
are derived from the spatial transcriptomics data of the slides (cell types via Tangram alignment to single-cell
reference atlases, followed by clustering and marker gene based refinement) and are NOT manual annotations,
see the publication for the quality control. The slides are cut into patches of 256x256 pixels with an overlap
of 64 pixels: the Hugging Face release used here contains 154,814 patches at 20x and 587,555 patches at 40x
magnification, extracted from 27 slides (see `SLIDES`).

Each patch is stored in a separate hdf5 file, which contains the RGB image ('image', channel first) and the labels
'labels/instances' (nucleus instance ids, unique per patch, 0 is background) and 'labels/semantic' (cell type of
each nucleus pixel). The cell types are the harmonized 'cells_final_label_group' of the per-slide metadata, with
the label ids given by `CLASS_NAMES` (1-based, 0 is background). Nuclei can be cut at the patch border, and
neighbouring patches overlap, so the same cell can appear in several patches.

NOTE: The data is stored in large parquet files, each holding the patches of one or several slides, and this
module converts them into the patch files once. The parquet files are kept next to the converted data. All slides
need 18 GB (20x) or 54 GB (40x) to download plus the converted patches. Use `slides` to only download and convert a
subset. This requires the pyarrow python package.

The data is located at https://huggingface.co/datasets/FelicieGS/STHELAR_20x and
https://huggingface.co/datasets/FelicieGS/STHELAR_40x, released under a CC-BY-4.0 license. The full data,
including the per-cell analyses, is available at https://doi.org/10.6019/S-BIAD2146.

This dataset is from the publication https://doi.org/10.1038/s41597-026-06937-6.
Please cite it if you use this dataset for your research.
"""

import os
import uuid
from io import BytesIO
from glob import glob
from tqdm import tqdm
from concurrent import futures
from natsort import natsorted
from typing import Union, Tuple, List, Literal, Optional, Sequence

import numpy as np
import imageio.v3 as imageio
from scipy.sparse import load_npz

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


CLASS_NAMES = [
    "Epithelial", "Blood_vessel", "Fibroblast_Myofibroblast", "Myeloid", "B_Plasma", "T_NK", "Melanocyte",
    "Specialized", "Other",
]
CLASS_IDS = {name: i for i, name in enumerate(CLASS_NAMES, start=1)}

MAGNIFICATIONS = ["20x", "40x"]
COMPLETE_MARKER = ".complete"

SLIDES = [
    "breast_s0", "breast_s1", "breast_s3", "breast_s6",
    "cervix_s0", "colon_s1", "colon_s2", "heart_s0",
    "kidney_s0", "kidney_s1", "liver_s0", "liver_s1",
    "lung_s1", "lung_s3", "lymph_node_s0", "ovary_s0",
    "ovary_s1", "pancreatic_s0", "pancreatic_s1", "pancreatic_s2",
    "prostate_s0", "skin_s1", "skin_s2", "skin_s3",
    "skin_s4", "tonsil_s0", "tonsil_s1",
]

REPOS = {"20x": "FelicieGS/STHELAR_20x", "40x": "FelicieGS/STHELAR_40x"}
REVISIONS = {
    "20x": "a01e22cdb7368edf595edf6150d0ac57deb5e548",
    "40x": "e32a8cdd50eff2d38e237f3729e9ac85bbb5203b",
}

SLIDE_SHARDS = {
    "20x": {
        "breast_s0": [0, 1],
        "breast_s1": [1, 2, 3, 4],
        "breast_s3": [4, 5, 6],
        "breast_s6": [6, 7],
        "cervix_s0": [7, 8],
        "colon_s1": [8, 9],
        "colon_s2": [9],
        "heart_s0": [9],
        "kidney_s0": [9],
        "kidney_s1": [9, 10],
        "liver_s0": [10],
        "liver_s1": [10, 11],
        "lung_s1": [11],
        "lung_s3": [11, 12],
        "lymph_node_s0": [12],
        "ovary_s0": [12],
        "ovary_s1": [12, 13],
        "pancreatic_s0": [13, 14],
        "pancreatic_s1": [14],
        "pancreatic_s2": [14],
        "prostate_s0": [14, 15],
        "skin_s1": [15],
        "skin_s2": [15, 16],
        "skin_s3": [16],
        "skin_s4": [16],
        "tonsil_s0": [16, 17],
        "tonsil_s1": [17],
    },
    "40x": {
        "breast_s0": [0, 1, 2, 3, 4],
        "breast_s1": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        "breast_s3": [15, 16, 17, 18],
        "breast_s6": [18, 19, 20, 21, 22],
        "cervix_s0": [22, 23, 24, 25, 26],
        "colon_s1": [26, 27],
        "colon_s2": [27, 28, 29],
        "heart_s0": [29],
        "kidney_s0": [29, 30],
        "kidney_s1": [30],
        "liver_s0": [30, 31, 32],
        "liver_s1": [32, 33],
        "lung_s1": [33, 34],
        "lung_s3": [34, 35, 36],
        "lymph_node_s0": [36, 37],
        "ovary_s0": [37, 38],
        "ovary_s1": [38, 39, 40, 41],
        "pancreatic_s0": [41, 42, 43],
        "pancreatic_s1": [43, 44],
        "pancreatic_s2": [44, 45],
        "prostate_s0": [45, 46],
        "skin_s1": [46, 47],
        "skin_s2": [47, 48],
        "skin_s3": [48, 49],
        "skin_s4": [49],
        "tonsil_s0": [49, 50, 51, 52],
        "tonsil_s1": [52, 53],
    },
}

SHARD_CHECKSUMS = {
    "20x": [
        "2beb26d148a5bd164289708794ebbcdb79b974135732509db8eb5d8a99e61840",
        "8c4a010b9049a64f93eb3bc819789a713640b815be0b7c7d84b0aff50093a86a",
        "f1bbcc6b5d3792554e376f5353b9ddc4168167ccb773823dd55bb3967d36cdf9",
        "e4642280e2a51c3175e2555f2110f88c3c3c448a4c1c0877f20f7b1780c5ba54",
        "8254d611a92ecd81e8e4f985b352f41767177dc6c941ff4f61e3f9499bce87bb",
        "64cf474fab05b96ced0cd0e14b429503975ee4e9af472f7d28d07c66c2f43390",
        "c45fff34cb07e2e28565a8a283b32c032c3c49a6d7b16f819d4970e405e4bcf6",
        "3201ca4493a73aaef48113127ad33f022c0e3aa878e3ad2935d21ba1f593e86b",
        "2bdfe7443d7247c89801ea6ad51f2a948b46fd022e1bf33614a8bfc3e9f6c657",
        "fd0be2332dcda2e95718a245e00943a2289d6fee940a70032d2b3a04bc2110cf",
        "88cdba926f9487d89e88b09926b2fa1a808bb9f839ef675080980ca262554dfe",
        "b3de85309f6b2f044b49ada57bbf2e9589293b1da81e436a0c615a945351759e",
        "cfd75a5137dba283bc2d14cc47f56b1055f30f84a8fc056714f6714fdde1137d",
        "2dd980ca2cb79ce746fafafdd6ff905c48014734d6bbca0750591a9d22a6a7f5",
        "3d6b02ceba429b91f122423505f79bcc2c4b7a81ea22dc6b0f63f6b195a2ab63",
        "051691b8a73b811c929c19cbcfd0611e36c3cfee585d8c837201e6651be9f5c9",
        "99d9d589bc21788516237414a136c76758a76099713f3eea08d9b6df034bedd8",
        "d17ba1032ff76fae8d8a9c8ba529facd073519b81423e686790ad6fba773c0eb",
    ],
    "40x": [
        "fded6f9036cdba4c0550e995d1dc2cd46f68ba14276419378d49c523a2ace915",
        "9090da789118603f02cc451ca4e28a4e0643c447e0a6f61e1f79925022a3a875",
        "c9ed431b0d63631e299c9fb909812b4a0f51160d4a780cb5f9e5611a3e1265b5",
        "3e548fa7f80d8ee8455fe399f15eb554954fa987de0348764c0c5f9dac97373c",
        "2cb12c3eab945479624e93658fc54e6ba441d3eaf279401f440fc2943291f319",
        "7c198b860fac69fc952fcc42a03ced84e414a4c6618a8322e4c4092ae93f2185",
        "742ff210dd58adc96e9759037811ddeefd865f042b01e8206e784045dc1868f6",
        "8263ebaee38fdce69d6a052c9488043b952136e93a8abe7c3852171c382c3cf2",
        "5b66f363aa6dd86d98eaed356ca16b53df0dcb6cb4ae6d9dc2c72f15ba38865c",
        "7d39f3cde0a3000b4ab6b0bfb9d845ba14cdb80574fd1f17cb64e0cb8e81a5e9",
        "90ede649a250ea7877d97b400bbe64f01506b96de5b6e420eb9a5ab08cebd279",
        "5e79e72a0fa406e0f070c65b47b07a9b390c9c712a055185e0ab34059fa2ed84",
        "df044c5bfc34be9e3d930d286fd7665e19810ed3d8a42dde7233b0deb8e5b8b3",
        "57baa4ee89d96e0b4712e874a680a7cd3ce4b92b38385576d9593aadcfa7bf7f",
        "db5d9801b45ef358d1c8b90865ebcef8660f4628ab5195a9d0651a8373dffada",
        "1cdd7d08ec094f4f950825af322518558928523a1e58c44e59b6e9d44f8ec6e1",
        "463c18d8cacd82e05738a00b6bd09bbfc1f53eab93c04abf78df0935afb6c4df",
        "b2a265d3a65d1a3c1461763fde7a088b4f6d3ed94f5ec6bb04e3debcb84201aa",
        "b62961040f0ba1e5ac56dc3a328a1b478001bee668e03b03fde4a75cc2cf7266",
        "354f00172234bd2e788b7874c31bab201eff84fafaf19ad425e2d8705b79b8d6",
        "90a20a79276eb571d8840ee3391923eca32d075c940926a674a51f8fa86a5cf8",
        "38601ae47fa7ac50a6327e466395acb82f20adb1a18314d578f11458f2d55574",
        "96fbf1edb41689439d7e327dab80a24a6b720f143054b35de1193d21c49cad14",
        "889e24ea440f9bbc0e086f91086e4f30a6a409c0aa4ab4004ec8887a7897e3ea",
        "38dc9cf1a2bd50c6b21e32f4fc2dfc630d0691acee6e17159ff2f741bcd36379",
        "19f8bd33957c0fa13e54f858a8ea422095ee4ad30708c9f872c66f81ec7f135a",
        "a478b13ae480f546f3f74df8af6e5e7be78a83d82574814d3e4994205e317ebb",
        "b7c6a826ebfa182c421c652fe850c2efc0d9e23090ea6378be09925297197862",
        "042eef94be71a40268f0fd12a88bded6d1d99ec08aace18233728823a61523de",
        "c1308467499672cd49110a7ddea1bb0ce0d009faba744da67a6da0e305d93bb5",
        "7ba700b6d595e7e6f72103392ebc16cd1cdf94c79174ae79765cf96768e8bddc",
        "ad928308055f7e86813ab613a05aa0c6bd65f78cfe7bab8892c38f90264fbba9",
        "780810e23adac20be6e5c8fd503e4261c5cf78dfd53ce3e36056294288a6dd44",
        "89f577a75a07026fb3293fcac5a9d451958232a0960866908c5282f9eaae5817",
        "8df8f7310cbba63e047e8ae9d30987e09f4697ae927cc7c7b7e08d22bcf093f6",
        "aaf8357b174e2a53532bcb213d3ba21e79dfa4c82cf39ebf338bb6351f1f1de6",
        "6301ba19059423d481e11617237c058f9b8382f4993401baa14ffc9732d6e3c6",
        "289e3dbbaaf77e06a57e4aa711f61f5eb4d7fb6b3f97247967874400db64ea57",
        "5d67d1601a292dc9f371f2ea3349036fde6bbcb18eaa035349c3600ec0dbeb31",
        "ec7b89e7acd33c9dfbc6b9e12737bcb6da887fd2ca8634cb330b32455c4c5833",
        "cacb6d40d9dd4e9fe0441c18c0d2ed687da738fc7c12c1767feaa96ad6a52f3d",
        "571a72273be1fa6f3c14e99087ba130f2f83b098f9674291e1f705f0e4e2752d",
        "1470a4745b0fe58253d7c484e4d6304e9fc3c008158f66fe403f6fb00e698ee4",
        "f33e597a2aba5eff981144c5eaa89d51844de76a1a9dcf58b22da7e9a6a301a4",
        "42e617f6f15ce018d2d11a5ba20527c97414aaea14debf28600240542cf55f9c",
        "000ecc178dab26dcca21a282d410b51b2b3bec8edb800edc60baa40fd34124b7",
        "2240e03cfd020c8a9e53524e335f17d1edb4b28319584726efd77429587e9547",
        "487f7a86152e7acfd623a27fade0fc0c88ad89367f764971bba1f619bff5e140",
        "af125205e4ff222db33611a2de2ef398e485f11ad2e492632ecb6524de0a0b07",
        "67ea31dc49c0022d803e143c41e79a61feca2bfb98d0e001c45523b33741d4eb",
        "d0883ae9c68b7eafad1ae9b8b75a3befcc1aaa86bb1120d022b90854e9d73ed8",
        "10285d36a9024b2a036b3f019f294dcc6a38d41b348a8176702a7ce87c587f02",
        "d28f98f0bf468f2fc14f95f1c721175513269ee9a18a13d4314c5e5a848d2e23",
        "94f2cea4dfd344933c942d14243744b882cc2c10d42628f95ff51c7253c28df7",
    ],
}

METADATA_CHECKSUMS = {
    "20x": {
        "breast_s0": "1cdb4bba93afb1c3d3eeca8f2caacb6a5d10b22f07b51acce274f8f015588bf5",
        "breast_s1": "abf1ebba18841e7a86f3114516a07574221f7d6978989c232dedcc7a258169e6",
        "breast_s3": "435e1a77a0acb2842d08276ad7e3cc67916a6e3949a56aca2260b831b5e0e26e",
        "breast_s6": "02b60f4bcb26646bc9273acd0dfc6ee98f6a6811266c7b862a625c2f5d87c15d",
        "cervix_s0": "3a198703ca5e228c81ea03166082316f857cc42416da5b303090206ca24abb4e",
        "colon_s1": "0f11618dae9209537855ce9568139181cc7d6223ea98810907ad299e77a0067e",
        "colon_s2": "823b0be5277fabf4930eae5d4402f4fabafd8876dc674c19e641b203a20a96b9",
        "heart_s0": "4f1a38ab08f83603faac675a3fba49fcc3528fda91a5c288b4efd5d2f34a878a",
        "kidney_s0": "a4db038e375a8f47532f689fd457516e761acf9c23cb90e128ac5d134b1b4f20",
        "kidney_s1": "a9bf51936b349456b836c80de6ba131852e91e3e040f20eda3802d837a27533c",
        "liver_s0": "17d9a987a88460824dabde15a55e5e864cf606aed57e2d3f2f8b4505b9c2d2eb",
        "liver_s1": "1e50bcda0c31cdd25a84bed02d6c548881a3f3e8a20b57d3daa4982dad505bc1",
        "lung_s1": "83b059126e8a1bac93eee91cd36ffda90e02aac406f241dc17b09963bf488463",
        "lung_s3": "639ded2b7c195a44ad7708fef31e4314505fdced5a8097e0e95a3ab5624cd361",
        "lymph_node_s0": "040c24a0bcd156ecbdfb13d476bae127bf4339cd189dae460ee42f7c4bb0242a",
        "ovary_s0": "b77a2d78eaff3a69bc4796ab9310c1f708b0368ceaaeb7a3196490e4bbc5160f",
        "ovary_s1": "2e66104150da680f19789a1e6a97b06f0e5710c83a2803a6754278838c8c59ad",
        "pancreatic_s0": "5b242492f5ddd965cfa8e6fa991ceea51b9d549fccfb56e91c121e46baba7a6f",
        "pancreatic_s1": "a1f37d4b57028f814be2f55ff31f74c87f1e82f482558071086ae79078d8dd0e",
        "pancreatic_s2": "f006c40cbf16a5a2ccf0782fe872f7150d42c5247fec5e93452ae74e132cac56",
        "prostate_s0": "e719328ca0239dc0ec5f252f2dea88e5c0e94595b7911f4759c1df935bfbd114",
        "skin_s1": "60354b20b2824abea0dfec2c70e4df2555329d6c2672091ce47e07a88f038af5",
        "skin_s2": "ef4cd89a2df0ef04f777a184667661f1266a0e4cb6a2fc633e57f84b1783f95c",
        "skin_s3": "150412b1a9ff4489756567f970a9663b5c894fd50093241316f5e72ef7008e49",
        "skin_s4": "df44e89d5efbdbfb5d3702e92921ebc38060ee5df02bbb8a87604467780013ea",
        "tonsil_s0": "8edd68ce0858c303386ed017264b89d7ffa90ba61dec9cc937ed849668394493",
        "tonsil_s1": "3dc4adb307b85ac7e53bcf65a8d77ce02b62f058d58fcab8543618e804f815e9",
    },
    "40x": {
        "breast_s0": "f8ddf4696e0a0dd927a8b0c1e24242d3f4575190c7a26fe80f78d0932bb23227",
        "breast_s1": "5fea33e2716a92054f02f3d8d75ca96a37f7e75acf91821451f6bd771d2fbc00",
        "breast_s3": "457a3fdabaad46f8e007958419a5828a6d50e438f236b1bb0064b8d604712730",
        "breast_s6": "64036da240ccf08273156939dbe1a3545951d07d1f62d97db7d3a3f7c8758689",
        "cervix_s0": "dc980ba27a43a6fe826e02cf529cf17e35f98a22781a530cd5c2b298ebb89d1c",
        "colon_s1": "27a31ef22fe716285f63dab79bf45aa9cdf6e8e860b8ca725910a239b800050c",
        "colon_s2": "ad1d3e2cfa20c39da3c4db03c4760732a0e4d851b3ef7f8b45f30cb6ba74756f",
        "heart_s0": "cfa6690e8be28714583edb866cb3fadf575733cdd084b6d3e8cc68e49b3917d8",
        "kidney_s0": "70ea019dc4c89647fe4ea34f65061b1386cac46cf69f78e036b1d3f14b6c1766",
        "kidney_s1": "88fd5484af6f5d1734a7f0f7a8fe6e50151a39c73c48a11892baf79ffd1f724e",
        "liver_s0": "b6f74e1ecfaa59bbb75921ba6855877f5692487e6890e7a396cd0684d7cb467d",
        "liver_s1": "a6058b2e2ed771e7cf82dfc1f22c3590fe99d003abdd22d18cbe9266f44bc03d",
        "lung_s1": "041fbcae0f18ae626c2b5daa37231020bad1edf589ea5149418ca9b727eae2c8",
        "lung_s3": "4c2ad9337adb0683604f0c8683288cb8eb902f095dc09eeb8764c58ab6c75386",
        "lymph_node_s0": "e1b4659526ee64a22eada40cec267a7a1b75fc3079d84f6123bfe547bf2716b2",
        "ovary_s0": "9e561f24a1cc9b6a3dfd9e2c2320c5d11a9ebae9d54bc5ddd6d9db3c446b5375",
        "ovary_s1": "79f100d35cedee540d5f5319549a381faf8ebde736601e329a99f815b3978980",
        "pancreatic_s0": "f154d79606c8d1acff94c032bfc975e8518913f488da2bc9224d8ae32162c05b",
        "pancreatic_s1": "e46c59592eea273450012d599dc09e87a970dab35eae1e9f00f431ecdd1201ed",
        "pancreatic_s2": "efccb9598ad581bcc0da9fd7c4b2b585b7c405774f0359e2380dd2ae51f56a54",
        "prostate_s0": "5397a52b30862f60780a8d105cacfa5a5ef7db66896edc88def2a6aa7b55cb3e",
        "skin_s1": "4b4faf5248e8555fd68395f16d7a6ca653c02f7b45171da224f77026e5246f8c",
        "skin_s2": "18b149fae149ac1b35c09701d8815b5818c90b5670643269a465d6aa34597b62",
        "skin_s3": "2cd2e0c9d49c56192cb2eec65d2f4bec3feb0fef3a986472a2c50113e61e5bd5",
        "skin_s4": "e5879ec0348b02eb1c6cb8ca04b7c33f88d50328e95008309453ef24e21e1702",
        "tonsil_s0": "0f95581fdc179715ac190df5e6a02df03e48baf156c4c700fc011101ba9315ff",
        "tonsil_s1": "5f40a96f6c11cdfba8f80f4eb0bdf14c0b34000166c5d116a3714bf25e478222",
    },
}


def _validate(magnification, slides):
    if magnification not in MAGNIFICATIONS:
        raise ValueError(f"'{magnification}' is not a valid magnification. Choose one of {MAGNIFICATIONS}.")
    if slides is None:
        return list(SLIDES)
    unknown = [s for s in slides if s not in SLIDES]
    if unknown:
        raise ValueError(f"{unknown} are not valid slides. Choose from {SLIDES}.")
    return [s for s in SLIDES if s in slides]


def _shard_name(magnification, shard_id):
    return f"train-{shard_id:05d}-of-{len(SHARD_CHECKSUMS[magnification]):05d}.parquet"


def _resolve_url(magnification, relative_path):
    return f"https://huggingface.co/datasets/{REPOS[magnification]}/resolve/{REVISIONS[magnification]}/{relative_path}"


def _load_lookup(metadata_path):
    import pandas as pd

    df = pd.read_parquet(metadata_path, columns=["cell_id_int", "cells_final_label_group"])
    codes = df["cells_final_label_group"].map(CLASS_IDS)
    assert not codes.isna().any(), f"Unknown cell types in {metadata_path}."
    ids = df["cell_id_int"].to_numpy().astype("int64")
    order = np.argsort(ids)
    return ids[order], codes.to_numpy().astype("uint8")[order]


def _write_patch(out_path, image_bytes, cell_id_bytes, lookup):
    if os.path.exists(out_path):
        return

    import h5py

    image = imageio.imread(image_bytes, extension=".png")
    assert image.ndim == 3 and image.shape[-1] == 3
    cell_ids = load_npz(BytesIO(cell_id_bytes)).toarray().astype("int64")
    assert cell_ids.shape == image.shape[:2]

    unique_ids, inverse = np.unique(cell_ids, return_inverse=True)
    instances = inverse.reshape(cell_ids.shape)
    if unique_ids[0] == 0:
        nucleus_ids = unique_ids[1:]
    else:
        nucleus_ids = unique_ids
        instances = instances + 1

    meta_ids, meta_codes = lookup
    positions = np.minimum(np.searchsorted(meta_ids, nucleus_ids), len(meta_ids) - 1)
    assert (meta_ids[positions] == nucleus_ids).all(), "Found cell ids without metadata."
    semantic = np.concatenate([[0], meta_codes[positions]]).astype("uint8")[instances]

    tmp_path = f"{out_path}.{uuid.uuid4().hex}.incomplete"
    with h5py.File(tmp_path, "w") as f:
        f.create_dataset("image", data=image.transpose((2, 0, 1)), compression="gzip")
        f.create_dataset("labels/instances", data=instances.astype("int32"), compression="gzip")
        f.create_dataset("labels/semantic", data=semantic, compression="gzip")
    os.replace(tmp_path, out_path)


def _convert_shard(shard_path, slides, lookups, preprocessed_dir, n_workers):
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(shard_path)
    with futures.ThreadPoolExecutor(n_workers) as pool:
        for row_group in tqdm(range(parquet_file.num_row_groups), desc=f"Convert {os.path.basename(shard_path)}"):
            table = parquet_file.read_row_group(row_group, columns=["file_name", "slide_id", "image", "cell_id_map"])
            slide_ids = table.column("slide_id").to_pylist()
            rows = [i for i, slide_id in enumerate(slide_ids) if slide_id in slides]
            if not rows:
                continue

            file_names = table.column("file_name").to_pylist()
            images = table.column("image").to_pylist()
            cell_id_maps = table.column("cell_id_map").to_pylist()
            tasks = []
            for i in rows:
                out_path = os.path.join(preprocessed_dir, slide_ids[i], f"{os.path.splitext(file_names[i])[0]}.h5")
                tasks.append(
                    pool.submit(_write_patch, out_path, images[i]["bytes"], cell_id_maps[i], lookups[slide_ids[i]])
                )
            for task in tasks:
                task.result()


def get_sthelar_data(
    path: Union[os.PathLike, str],
    magnification: Literal["20x", "40x"] = "20x",
    slides: Optional[Sequence[str]] = None,
    download: bool = False,
) -> str:
    """Download the STHELAR dataset and convert its patches to hdf5 files.

    NOTE: All slides need 18 GB (20x) or 54 GB (40x) for the download. Use `slides` for a subset. The parquet files
    of the slides are kept, since a parquet file can contain the patches of several slides.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        magnification: The choice of magnification. Either '20x' or '40x'.
        slides: The slides to use, see `SLIDES` for the valid choices. By default all slides are used.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the converted patches are stored, in one sub-folder per slide.
    """
    slides = _validate(magnification, slides)
    data_dir = os.path.join(path, magnification)
    preprocessed_dir = os.path.join(data_dir, "preprocessed")

    pending = [s for s in slides if not os.path.exists(os.path.join(preprocessed_dir, s, COMPLETE_MARKER))]
    if not pending:
        return preprocessed_dir

    os.makedirs(os.path.join(data_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "cell_metadata"), exist_ok=True)

    lookups = {}
    for slide in pending:
        relative_path = f"cell_metadata/{slide}_cell_metadata.parquet"
        metadata_path = os.path.join(data_dir, relative_path)
        util.download_source(
            path=metadata_path, url=_resolve_url(magnification, relative_path), download=download,
            checksum=METADATA_CHECKSUMS[magnification][slide],
        )
        lookups[slide] = _load_lookup(metadata_path)
        os.makedirs(os.path.join(preprocessed_dir, slide), exist_ok=True)

    n_cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    n_workers = min(8, n_cpus)

    shard_ids = sorted({i for slide in pending for i in SLIDE_SHARDS[magnification][slide]})
    for shard_id in shard_ids:
        relative_path = f"data/{_shard_name(magnification, shard_id)}"
        shard_path = os.path.join(data_dir, relative_path)
        util.download_source(
            path=shard_path, url=_resolve_url(magnification, relative_path), download=download,
            checksum=SHARD_CHECKSUMS[magnification][shard_id],
        )
        _convert_shard(shard_path, set(pending), lookups, preprocessed_dir, n_workers)

    for slide in pending:
        with open(os.path.join(preprocessed_dir, slide, COMPLETE_MARKER), "w"):
            pass

    return preprocessed_dir


def get_sthelar_paths(
    path: Union[os.PathLike, str],
    magnification: Literal["20x", "40x"] = "20x",
    slides: Optional[Sequence[str]] = None,
    download: bool = False,
) -> List[str]:
    """Get paths to the STHELAR data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        magnification: The choice of magnification. Either '20x' or '40x'.
        slides: The slides to use, see `SLIDES` for the valid choices. By default all slides are used.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the hdf5 files, which contain the image ('image') and the labels
        ('labels/instances' and 'labels/semantic') of a patch.
    """
    slides = _validate(magnification, slides)
    preprocessed_dir = get_sthelar_data(path, magnification, slides, download)

    data_paths = []
    for slide in slides:
        data_paths.extend(natsorted(glob(os.path.join(preprocessed_dir, slide, "*.h5"))))

    assert len(data_paths) > 0
    return data_paths


def get_sthelar_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    magnification: Literal["20x", "40x"] = "20x",
    slides: Optional[Sequence[str]] = None,
    label_type: Literal["instances", "semantic"] = "instances",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the STHELAR dataset for nucleus segmentation and cell type classification.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        magnification: The choice of magnification. Either '20x' or '40x'.
        slides: The slides to use, see `SLIDES` for the valid choices. By default all slides are used.
        label_type: The choice of labels. Either 'instances' for the nucleus instances or 'semantic' for the
            cell types of the nuclei, with the ids given by `CLASS_NAMES`.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    if label_type not in ("instances", "semantic"):
        raise ValueError(f"'{label_type}' is not a valid label type. Choose either 'instances' or 'semantic'.")

    data_paths = get_sthelar_paths(path, magnification, slides, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=data_paths,
        raw_key="image",
        label_paths=data_paths,
        label_key=f"labels/{label_type}",
        patch_shape=patch_shape,
        ndim=2,
        with_channels=True,
        **kwargs
    )


def get_sthelar_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    magnification: Literal["20x", "40x"] = "20x",
    slides: Optional[Sequence[str]] = None,
    label_type: Literal["instances", "semantic"] = "instances",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the STHELAR dataloader for nucleus segmentation and cell type classification.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        magnification: The choice of magnification. Either '20x' or '40x'.
        slides: The slides to use, see `SLIDES` for the valid choices. By default all slides are used.
        label_type: The choice of labels. Either 'instances' for the nucleus instances or 'semantic' for the
            cell types of the nuclei, with the ids given by `CLASS_NAMES`.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_sthelar_dataset(
        path, patch_shape, magnification, slides, label_type, resize_inputs, download, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
