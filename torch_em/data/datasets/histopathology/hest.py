"""The HEST-1k dataset contains 1,276 paired H&E whole-slide images and spatial transcriptomics
profiles across 26 organ types. For each sample, pre-extracted 224x224 H&E patches at 0.5 um/px,
CellViT nuclei instance segmentation masks, Xenium DAPI-derived nucleus boundaries (for Xenium
samples), and cell-level spatial transcriptomics gene expression profiles are available.

Three types of segmentation labels are supported:
- 'instances': nuclei instance masks derived from CellViT (H&E-based, all samples).
- 'xenium_instances': nuclei instance masks from DAPI segmentation (Xenium samples only).
- 'semantic': cell-type semantic masks derived from spatial transcriptomics via Leiden clustering
  and PanglaoDB marker-gene voting (Xenium samples only). Classes: 0=background, 1=Epithelial,
  2=Inflammatory, 3=Connective, 4=Neoplastic, 5=Unknown.

This dataset is used in the paper https://doi.org/10.48550/arXiv.2604.23481 as a scalable
alternative to manually annotated datasets for nuclei segmentation and classification training.

The dataset is located at https://huggingface.co/datasets/MahmoodLab/hest.
This dataset is from the following publication:
- Jaume et al. (2024): https://doi.org/10.48550/arXiv.2406.16192
Please cite it if you use this dataset in your research.

NOTE: Requires huggingface_hub for download: pip install huggingface_hub
NOTE: Requires geopandas, rasterio, and scipy for preprocessing: pip install geopandas rasterio scipy
NOTE: Requires scanpy, python-igraph, and leidenalg for semantic labels: pip install scanpy igraph leidenalg
NOTE: The full dataset is ~2 TB. Use the `organs` argument to download only a subset.
"""

import json
import os
import zipfile
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import torch_em


HF_REPO = "MahmoodLab/hest"
METADATA_FILENAME = "HEST_v1_3_0.csv"
PANGLAODB_URL = "https://panglaodb.se/markers/PanglaoDB_markers_27_Mar_2020.tsv.gz"

# Integer label for each cell-type category (0 = background).
CELL_TYPE_LABELS = {"Epithelial": 1, "Inflammatory": 2, "Connective": 3, "Neoplastic": 4, "Unknown": 5}

# Map from public label_choice strings to HDF5 dataset paths.
LABEL_KEYS = {
    "instances": "labels/instances/h&e",
    "xenium_instances": "labels/instances/xenium",
    "semantic": "labels/semantic/st",
}

# Organs present in both HEST-1k and PanNuke (used in arXiv 2604.23481).
PANNUKE_ORGANS = [
    "Breast", "Colon", "Kidney", "Liver", "Lung", "Ovarian", "Pancreatic", "Prostate", "Skin", "Stomach",
]

# Keyword fragments for mapping PanglaoDB cell type names to coarse categories.
EPITHELIAL_KEYWORDS = [
    "acinar", "airway epithelial", "airway goblet", "alveolar type", "alpha cell", "basal cell",
    "beta cell", "cholangiocyte", "ciliated", "clara", "crypt", "delta cell", "ductal",
    "enterocyte", "epithelial", "goblet", "hepatocyte", "keratinocyte", "mesothelial",
    "paneth", "pneumocyte", "proximal tubule", "renal tubule", "squamous", "thyroid",
    "trophoblast", "tuft", "urothelial",
]
INFLAMMATORY_KEYWORDS = [
    "alveolar macrophage", "b cell", "basophil", "dendritic", "eosinophil",
    "innate lymphoid", "lymphocyte", "macrophage", "mast cell", "monocyte",
    "natural killer", "neutrophil", "nk cell", "plasma cell", "regulatory t", "t cell",
]
CONNECTIVE_KEYWORDS = [
    "adipocyte", "chondrocyte", "endothelial", "fibroblast", "mesenchymal",
    "myofibroblast", "osteoblast", "osteoclast", "pericyte", "smooth muscle",
    "stellate", "stromal", "vascular",
]

# Well-known cancer-associated genes (COSMIC Cancer Gene Census, tier 1).
CANCER_GENES = {
    "ABL1", "AKT1", "ALK", "APC", "ATM", "BRAF", "BRCA1", "BRCA2", "CDH1", "CDKN2A",
    "CTNNB1", "EGFR", "ERBB2", "ESR1", "EZH2", "FBXW7", "FGFR1", "FGFR2", "FGFR3",
    "FLT3", "GATA3", "GNAQ", "GNAS", "HNF1A", "HRAS", "IDH1", "IDH2", "JAK2", "KIT",
    "KRAS", "MAP2K1", "MDM2", "MET", "MLH1", "MSH2", "MSH6", "MTOR", "MYC", "MYCN",
    "NF1", "NF2", "NFE2L2", "NOTCH1", "NOTCH2", "NRAS", "PALB2", "PBRM1", "PIK3CA",
    "PIK3R1", "PMS2", "POLE", "PTCH1", "PTEN", "RB1", "RET", "RNF43", "SETD2", "SF3B1",
    "SMAD4", "SMARCA4", "SMARCB1", "SMO", "STK11", "TERT", "TET2", "TP53", "TSC1",
    "TSC2", "VHL", "BAP1", "CDK12", "CHEK2", "CREBBP", "DNMT3A", "EP300", "FANCD2",
    "KDM5C", "KDM6A", "KEAP1", "MAP3K1", "MUTYH", "NBN", "PDGFRA", "PPP2R1A", "RAD51C",
    "RUNX1", "SDHA", "SDHB", "SDHC", "SDHD", "SUFU", "TP63", "XRCC2", "AXIN1", "AXIN2",
    "BRIP1", "CHD4", "ELOC", "FANCA", "FH", "FLCN", "MRE11", "RAD50", "RAD51B", "RAD51D",
}


def _download_hest(path: str, sample_ids: List[str], include_xenium: bool, include_st: bool) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError("huggingface_hub is required. Install with: pip install huggingface_hub")

    patterns = [METADATA_FILENAME]
    for sid in sample_ids:
        patterns += [f"patches/{sid}.h5", f"cellvit_seg/{sid}_cellvit_seg.geojson.zip"]
        if include_xenium:
            patterns += [f"xenium_seg/{sid}_xenium_nucleus_seg.parquet"]
        if include_st:
            patterns += [f"st/{sid}.h5ad"]

    os.makedirs(path, exist_ok=True)
    snapshot_download(repo_id=HF_REPO, repo_type="dataset", local_dir=path, allow_patterns=patterns)


def _load_metadata(path: str) -> "pd.DataFrame":  # noqa
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required. Install with: pip install pandas")

    csv_path = os.path.join(path, METADATA_FILENAME)
    if not os.path.exists(csv_path):
        raise RuntimeError(f"Metadata not found at {csv_path}. Run get_hest_data() first.")
    return pd.read_csv(csv_path)


def _filter_sample_ids(path: str, organs: Optional[List[str]]) -> List[str]:
    meta = _load_metadata(path)
    if organs is not None:
        meta = meta[meta["organ"].isin(organs)]
    return meta["id"].tolist()


def _unzip_cellvit(zip_path: str, out_dir: str) -> Optional[str]:
    if not os.path.exists(zip_path):
        return None
    # Strip both extensions: "SAMPLEID_cellvit_seg.geojson.zip" -> "SAMPLEID"
    sample_id = os.path.basename(zip_path).replace("_cellvit_seg.geojson.zip", "")
    extract_dir = os.path.join(out_dir, sample_id)
    if not os.path.exists(extract_dir):
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
    matches = glob(os.path.join(extract_dir, "**", "*.geojson"), recursive=True)
    return matches[0] if matches else None


def _gdf_from_xenium_parquet(parquet_path: str) -> "gpd.GeoDataFrame":  # noqa
    """Load a Xenium nucleus segmentation parquet into a GeoDataFrame.

    Expected format: index is cell_id, single 'geometry' column with WKB-encoded polygons.
    """
    try:
        import pandas as pd
        import geopandas as gpd
        import shapely
    except ImportError:
        raise ImportError("geopandas and shapely are required. Install with: pip install geopandas rasterio")

    df = pd.read_parquet(parquet_path)
    geometries = shapely.from_wkb(df["geometry"].values)
    return gpd.GeoDataFrame({"cell_id": df.index.astype(str), "geometry": geometries}, geometry="geometry")


def _gdf_from_cellvit_geojson(geojson_path: str) -> "gpd.GeoDataFrame":  # noqa
    """Load CellViT segmentation GeoJSON into a GeoDataFrame with one row per nucleus.

    The file is a JSON list of features with MultiPolygon geometries (one per cell-type class).
    Each MultiPolygon is exploded into individual Polygon rows.
    """
    try:
        import geopandas as gpd
        from shapely.geometry import shape, MultiPolygon
    except ImportError:
        raise ImportError("geopandas and shapely are required. Install with: pip install geopandas rasterio")

    with open(geojson_path) as fh:
        data = json.load(fh)

    records = []
    for feat in data:
        geom = shape(feat["geometry"])
        if isinstance(geom, MultiPolygon):
            for poly in geom.geoms:
                records.append({"geometry": poly})
        else:
            records.append({"geometry": geom})

    if not records:
        return gpd.GeoDataFrame(columns=["geometry"])
    return gpd.GeoDataFrame(records, geometry="geometry")


def _rasterize_patch_instances(
    patch_x: int,
    patch_y: int,
    patch_size: int,
    cells_gdf: "gpd.GeoDataFrame",  # noqa
    native_scale: float = 1.0,
) -> np.ndarray:
    """Rasterize nucleus polygons within one patch to an instance mask.

    native_scale: native WSI pixels per 0.5 um/px patch pixel (= 0.5 / pixel_size_um).
    Patches are stored at 0.5 um/px but cell coords are in native WSI pixel space.
    """
    try:
        from shapely.geometry import box
        from shapely.affinity import translate, scale as affine_scale
        from rasterio.features import rasterize as rio_rasterize
    except ImportError:
        raise ImportError("rasterio and shapely are required. Install with: pip install geopandas rasterio")

    native_size = round(patch_size * native_scale)
    patch_box = box(patch_x, patch_y, patch_x + native_size, patch_y + native_size)
    local = cells_gdf[cells_gdf.geometry.intersects(patch_box)].copy()
    mask = np.zeros((patch_size, patch_size), dtype=np.int32)
    if local.empty:
        return mask

    inv = 1.0 / native_scale
    local["geometry"] = local["geometry"].apply(
        lambda g: affine_scale(translate(g, xoff=-patch_x, yoff=-patch_y), xfact=inv, yfact=inv, origin=(0, 0))
    )
    shapes = ((geom, i + 1) for i, geom in enumerate(local.geometry))
    return rio_rasterize(shapes, out_shape=(patch_size, patch_size), fill=0, dtype=np.int32)


def _rasterize_patch_semantic(
    patch_x: int,
    patch_y: int,
    patch_size: int,
    cells_gdf: "gpd.GeoDataFrame",  # noqa
    spot_labels: np.ndarray,
    native_scale: float = 1.0,
    spot_tree=None,
) -> np.ndarray:
    """Rasterize nucleus polygons within one patch to a semantic (cell-type) mask.

    native_scale: native WSI pixels per 0.5 um/px patch pixel (= 0.5 / pixel_size_um).
    spot_labels: (N, 3) array of (x, y, label) for each ST spot in native WSI coordinates.
    spot_tree: pre-built cKDTree over spot_labels[:, :2]. Built locally if None (slow per-patch).
    Each nucleus is assigned the label of its nearest ST spot via KDTree.
    """
    try:
        from shapely.geometry import box
        from shapely.affinity import translate, scale as affine_scale
        from rasterio.features import rasterize as rio_rasterize
        from scipy.spatial import cKDTree
    except ImportError:
        raise ImportError("rasterio, shapely, and scipy are required. Install with: pip install geopandas rasterio scipy")  # noqa

    native_size = round(patch_size * native_scale)
    patch_box = box(patch_x, patch_y, patch_x + native_size, patch_y + native_size)
    local = cells_gdf[cells_gdf.geometry.intersects(patch_box)].copy()
    mask = np.zeros((patch_size, patch_size), dtype=np.int32)
    if local.empty:
        return mask

    # Assign each nucleus its nearest ST spot's label via KDTree on native coords.
    tree = spot_tree if spot_tree is not None else cKDTree(spot_labels[:, :2])
    centroids = np.array([[g.centroid.x, g.centroid.y] for g in local.geometry])
    _, idx = tree.query(centroids)
    local["label"] = spot_labels[idx, 2].astype(int)

    inv = 1.0 / native_scale
    local["geometry"] = local["geometry"].apply(
        lambda g: affine_scale(translate(g, xoff=-patch_x, yoff=-patch_y), xfact=inv, yfact=inv, origin=(0, 0))
    )
    shapes = ((geom, int(label)) for geom, label in zip(local.geometry, local["label"]))
    return rio_rasterize(shapes, out_shape=(patch_size, patch_size), fill=0, dtype=np.int32)


def _load_panglaodb(cache_path: str) -> "pd.DataFrame":  # noqa
    """Download (once) and return the PanglaoDB marker-gene table."""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required. Install with: pip install pandas")

    tsv_path = os.path.join(cache_path, "PanglaoDB_markers.tsv.gz")
    if not os.path.exists(tsv_path):
        import urllib.request
        os.makedirs(cache_path, exist_ok=True)
        req = urllib.request.Request(PANGLAODB_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(tsv_path, "wb") as fh:
            fh.write(resp.read())

    df = pd.read_csv(tsv_path, sep="\t")
    # Keep only human genes.
    df = df[df["species"].str.contains("Hs", na=False)]
    return df[["official gene symbol", "cell type"]].copy()


def _cell_type_to_category(cell_type_name: str) -> str:
    """Map a PanglaoDB cell type name to one of the four coarse categories."""
    name = cell_type_name.lower()
    for kw in EPITHELIAL_KEYWORDS:
        if kw in name:
            return "Epithelial"
    for kw in INFLAMMATORY_KEYWORDS:
        if kw in name:
            return "Inflammatory"
    for kw in CONNECTIVE_KEYWORDS:
        if kw in name:
            return "Connective"
    return "Unknown"


def _compute_cell_type_map(
    h5ad_path: str,
    marker_db: "pd.DataFrame",  # noqa
    top_n: int = 10,
    tau_vote: int = 5,
    top_m: int = 20,
    tau_cancer: float = 0.25,
) -> np.ndarray:
    """Run the ST cell-type assignment pipeline from the paper (arXiv 2604.23481).

    Returns an (N, 3) float32 array of (x, y, label) for each ST spot, where x and y
    are native WSI pixel coordinates (pxl_col_in_fullres / pxl_row_in_fullres). Callers
    use a KDTree to assign each segmented nucleus to its nearest ST spot.
    """
    try:
        import scanpy as sc
    except ImportError:
        raise ImportError("scanpy is required for semantic labels. Install with: pip install scanpy")

    adata = sc.read_h5ad(h5ad_path)

    if "pxl_col_in_fullres" not in adata.obs.columns or "pxl_row_in_fullres" not in adata.obs.columns:
        raise ValueError("h5ad missing pxl_col_in_fullres / pxl_row_in_fullres spot coordinates.")

    # Build gene -> category lookup from PanglaoDB.
    gene_to_cats: Dict[str, List[str]] = {}
    for gene, ct in zip(marker_db["official gene symbol"], marker_db["cell type"]):
        cat = _cell_type_to_category(ct)
        gene_to_cats.setdefault(gene, []).append(cat)

    # Preprocessing and clustering.
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.leiden(adata, resolution=4.0)
    sc.tl.rank_genes_groups(adata, groupby="leiden", method="wilcoxon")

    cluster_cat: Dict[str, str] = {}
    for cluster in adata.obs["leiden"].unique():
        try:
            top_genes = list(
                sc.get.rank_genes_groups_df(adata, group=cluster)["names"].iloc[:top_m]
            )
        except Exception:
            cluster_cat[cluster] = "Unknown"
            continue

        votes: Dict[str, float] = {"Epithelial": 0.0, "Inflammatory": 0.0, "Connective": 0.0}
        total_vote = 0.0
        for rank, gene in enumerate(top_genes[:top_n]):
            weight = top_n - rank
            for cat in gene_to_cats.get(gene, []):
                if cat in votes:
                    votes[cat] += weight
                    total_vote += weight

        if total_vote < tau_vote:
            cluster_cat[cluster] = "Unknown"
            continue

        best = max(votes, key=votes.get)  # type: ignore[arg-type]
        cluster_cat[cluster] = best

        if best == "Epithelial":
            cancer_overlap = sum(1 for g in top_genes[:top_m] if g in CANCER_GENES)
            if cancer_overlap / top_m > tau_cancer:
                cluster_cat[cluster] = "Neoplastic"

    # Build (N, 3) array: (x, y, label) per ST spot in native WSI pixel coords.
    xs = adata.obs["pxl_col_in_fullres"].values.astype(np.float32)
    ys = adata.obs["pxl_row_in_fullres"].values.astype(np.float32)
    labels = np.array(
        [CELL_TYPE_LABELS[cluster_cat.get(adata.obs["leiden"].iloc[i], "Unknown")]
         for i in range(adata.n_obs)],
        dtype=np.float32,
    )
    return np.stack([xs, ys, labels], axis=1)


def _preprocess_sample(
    patches_h5: str,
    cellvit_geojson: Optional[str],
    xenium_parquet: Optional[str],
    h5ad_path: Optional[str],
    marker_db: Optional["pd.DataFrame"],  # noqa
    out_h5: str,
    patch_size: int = 224,
    pixel_size_um: float = 0.5,
) -> bool:
    # Cell coords are in native WSI pixel space; patches are at 0.5 um/px.
    # native_scale = native WSI pixels per 0.5 um/px patch pixel.
    native_scale = 0.5 / pixel_size_um

    with h5py.File(patches_h5, "r") as f:
        img_key = "img" if "img" in f else ("imgs" if "imgs" in f else "images")
        imgs = f[img_key][:]  # (N, H, W, 3) uint8
        coords = f["coords"][:]  # (N, 2) top-left (x, y) in native WSI pixels

    n = len(imgs)
    if n == 0:
        return False

    # Load GeoDataFrames once per slide.
    cellvit_gdf = None
    if cellvit_geojson is not None and os.path.exists(cellvit_geojson):
        cellvit_gdf = _gdf_from_cellvit_geojson(cellvit_geojson)

    xenium_gdf = None
    if xenium_parquet is not None and os.path.exists(xenium_parquet):
        xenium_gdf = _gdf_from_xenium_parquet(xenium_parquet)

    spot_labels: Optional[np.ndarray] = None
    if h5ad_path is not None and os.path.exists(h5ad_path) and marker_db is not None and xenium_gdf is not None:
        try:
            spot_labels = _compute_cell_type_map(h5ad_path, marker_db)
        except Exception as e:
            print(f"Warning: semantic labels unavailable for {os.path.basename(h5ad_path)}: {e}")

    # Build the KDTree once per slide rather than once per patch.
    spot_tree = None
    if spot_labels is not None:
        try:
            from scipy.spatial import cKDTree
            spot_tree = cKDTree(spot_labels[:, :2])
        except ImportError:
            pass

    raw = np.zeros((n, 3, patch_size, patch_size), dtype=np.uint8)
    instances = np.zeros((n, patch_size, patch_size), dtype=np.int32)
    xenium_instances = np.zeros((n, patch_size, patch_size), dtype=np.int32)
    semantic = np.zeros((n, patch_size, patch_size), dtype=np.int32)

    sid = os.path.splitext(os.path.basename(out_h5))[0]
    for i, (img, coord) in enumerate(tqdm(zip(imgs, coords), total=n, desc=f"Processing {sid}", leave=False)):
        raw[i] = img[:patch_size, :patch_size, :].transpose(2, 0, 1)
        px, py = int(coord[0]), int(coord[1])

        if cellvit_gdf is not None:
            instances[i] = _rasterize_patch_instances(px, py, patch_size, cellvit_gdf, native_scale)

        if xenium_gdf is not None:
            xenium_instances[i] = _rasterize_patch_instances(px, py, patch_size, xenium_gdf, native_scale)

        if spot_labels is not None and xenium_gdf is not None:
            semantic[i] = _rasterize_patch_semantic(
                px, py, patch_size, xenium_gdf, spot_labels, native_scale, spot_tree
            )

    chunk_2d = (1, patch_size, patch_size)
    with h5py.File(out_h5, "w") as f:
        f.create_dataset("raw", data=raw, compression="gzip", chunks=(1, 3, patch_size, patch_size))
        f.create_dataset(LABEL_KEYS["instances"], data=instances, compression="gzip", chunks=chunk_2d)
        f.create_dataset(LABEL_KEYS["xenium_instances"], data=xenium_instances, compression="gzip", chunks=chunk_2d)
        f.create_dataset(LABEL_KEYS["semantic"], data=semantic, compression="gzip", chunks=chunk_2d)

    return True


class HESTDataset(Dataset):
    """2D patch dataset for HEST-1k.

    Indexes all patches across all per-slide H5 files and returns proper 2D tensors:
    raw (3, H, W) float32 in [0, 1] and labels (H, W) int32.
    """

    def __init__(
        self,
        h5_paths: List[str],
        label_key: str,
        raw_transform: Optional[Callable] = None,
        label_transform: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        n_samples: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        self._label_key = label_key
        self._raw_transform = raw_transform
        self._label_transform = label_transform
        self._transform = transform

        # Build flat index: list of (h5_path, patch_idx).
        self._index: List[Tuple[str, int]] = []
        for h5_path in h5_paths:
            with h5py.File(h5_path, "r") as f:
                n = f["raw"].shape[0]  # raw stored as (N, 3, H, W)
            self._index.extend((h5_path, i) for i in range(n))

        if n_samples is not None:
            rng = np.random.default_rng(seed)
            chosen = rng.choice(len(self._index), size=n_samples, replace=n_samples > len(self._index))
            self._index = [self._index[i] for i in chosen]

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        h5_path, patch_idx = self._index[idx]
        with h5py.File(h5_path, "r") as f:
            raw = f["raw"][patch_idx].astype(np.float32) / 255.0  # (3, H, W)
            label = f[self._label_key][patch_idx].astype(np.int32)  # (H, W)

        raw = torch.from_numpy(raw)
        label = torch.from_numpy(label)

        if self._raw_transform is not None:
            raw = self._raw_transform(raw)
        if self._label_transform is not None:
            label = self._label_transform(label)
        if self._transform is not None:
            raw, label = self._transform(raw, label)

        return raw, label


def get_hest_data(
    path: Union[os.PathLike, str],
    organs: Optional[List[str]] = None,
    download: bool = False,
) -> str:
    """Download and preprocess the HEST-1k dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        organs: List of organ types to include. Uses all available organs if None.
            Example: ['Breast', 'Colon']. See PANNUKE_ORGANS for the set used in arXiv 2604.23481.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the preprocessed data directory.
    """
    preprocessed_dir = os.path.join(path, "preprocessed")

    if download:
        meta_path = os.path.join(path, METADATA_FILENAME)
        if not os.path.exists(meta_path):
            try:
                from huggingface_hub import hf_hub_download
            except ImportError:
                raise ImportError("huggingface_hub is required. Install with: pip install huggingface_hub")
            hf_hub_download(repo_id=HF_REPO, repo_type="dataset", filename=METADATA_FILENAME, local_dir=path)

        sample_ids = _filter_sample_ids(path, organs)
        xenium_dir = os.path.join(path, "xenium_seg")
        st_dir = os.path.join(path, "st")
        include_xenium = not os.path.exists(xenium_dir)
        include_st = not os.path.exists(st_dir)
        _download_hest(path, sample_ids, include_xenium=include_xenium, include_st=include_st)
    else:
        sample_ids = [
            os.path.splitext(os.path.basename(p))[0]
            for p in glob(os.path.join(path, "patches", "*.h5"))
        ]
        if organs is not None:
            meta_path = os.path.join(path, METADATA_FILENAME)
            if os.path.exists(meta_path):
                allowed = set(_filter_sample_ids(path, organs))
                sample_ids = [s for s in sample_ids if s in allowed]

    # Load PanglaoDB once for all samples.
    db_cache = os.path.join(path, "_db_cache")
    try:
        marker_db = _load_panglaodb(db_cache)
    except Exception:
        marker_db = None

    # Build a pixel_size lookup from the metadata (um/px at native resolution).
    try:
        meta = _load_metadata(path)
        pixel_size_map = dict(zip(meta["id"], meta["pixel_size_um_estimated"].fillna(0.5)))
    except Exception:
        pixel_size_map = {}

    os.makedirs(preprocessed_dir, exist_ok=True)
    cellvit_zip_dir = os.path.join(path, "cellvit_seg")
    cellvit_cache = os.path.join(path, "_cellvit_extracted")
    xenium_dir = os.path.join(path, "xenium_seg")
    st_dir = os.path.join(path, "st")

    for sid in tqdm(sample_ids, desc="Preprocessing HEST samples"):
        out_h5 = os.path.join(preprocessed_dir, f"{sid}.h5")
        if os.path.exists(out_h5):
            continue

        patches_h5 = os.path.join(path, "patches", f"{sid}.h5")
        if not os.path.exists(patches_h5):
            continue

        geojson_path = _unzip_cellvit(
            os.path.join(cellvit_zip_dir, f"{sid}_cellvit_seg.geojson.zip"), cellvit_cache
        )
        xenium_parquet = os.path.join(xenium_dir, f"{sid}_xenium_nucleus_seg.parquet")
        h5ad_path = os.path.join(st_dir, f"{sid}.h5ad")
        pixel_size_um = float(pixel_size_map.get(sid, 0.5))

        _preprocess_sample(
            patches_h5=patches_h5,
            cellvit_geojson=geojson_path,
            xenium_parquet=xenium_parquet if os.path.exists(xenium_parquet) else None,
            h5ad_path=h5ad_path if os.path.exists(h5ad_path) else None,
            marker_db=marker_db,
            out_h5=out_h5,
            pixel_size_um=pixel_size_um,
        )

    return preprocessed_dir


def get_hest_paths(
    path: Union[os.PathLike, str],
    organs: Optional[List[str]] = None,
    download: bool = False,
) -> List[str]:
    """Get paths to the preprocessed HEST-1k H5 files.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        organs: List of organ types to include. Uses all available organs if None.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths to the preprocessed H5 files (one per slide).
    """
    preprocessed_dir = get_hest_data(path, organs, download)
    h5_paths = natsorted(glob(os.path.join(preprocessed_dir, "*.h5")))
    if not h5_paths:
        raise RuntimeError(f"No preprocessed data found in {preprocessed_dir}.")

    if organs is not None:
        meta_path = os.path.join(path, METADATA_FILENAME)
        if os.path.exists(meta_path):
            allowed = set(_filter_sample_ids(path, organs))
            h5_paths = [p for p in h5_paths if os.path.splitext(os.path.basename(p))[0] in allowed]

    return h5_paths


def get_hest_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    organs: Optional[List[str]] = None,
    label_choice: Literal["instances", "xenium_instances", "semantic"] = "instances",
    download: bool = False,
    n_samples: Optional[int] = None,
    seed: Optional[int] = None,
    raw_transform: Optional[Callable] = None,
    label_transform: Optional[Callable] = None,
    transform: Optional[Callable] = None,
) -> Dataset:
    """Get the HEST-1k dataset for nuclei segmentation and cell-type classification.

    Returns a 2D dataset: each item is raw (3, H, W) float32 in [0, 1] and labels (H, W) int32.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: Not used for cropping (patches are already 224x224); kept for API consistency.
        organs: List of organ types to include. Uses all available organs if None.
            Use PANNUKE_ORGANS for the 10-organ subset from arXiv 2604.23481.
        label_choice: Which label type to return:
            - 'instances': CellViT nuclei instance masks (H&E-based, all samples).
            - 'xenium_instances': DAPI nuclei instance masks (Xenium samples only, zeros otherwise).
            - 'semantic': ST-derived cell-type labels 1-5 (Xenium samples only, zeros otherwise).
        download: Whether to download the data if it is not present.
        n_samples: Number of patches to sample (with replacement if larger than total). Uses all if None.
        seed: Random seed for reproducible patch sampling when n_samples is set.
        raw_transform: Transform applied to the raw image tensor.
        label_transform: Transform applied to the label tensor.
        transform: Joint transform applied to both raw and label.

    Returns:
        The segmentation dataset.
    """
    valid = ("instances", "xenium_instances", "semantic")
    if label_choice not in valid:
        raise ValueError(f"'{label_choice}' is not valid. Choose from {valid}.")

    h5_paths = get_hest_paths(path, organs, download)
    return HESTDataset(
        h5_paths=h5_paths,
        label_key=LABEL_KEYS[label_choice],
        raw_transform=raw_transform,
        label_transform=label_transform,
        transform=transform,
        n_samples=n_samples,
        seed=seed,
    )


def get_hest_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    organs: Optional[List[str]] = None,
    label_choice: Literal["instances", "xenium_instances", "semantic"] = "instances",
    download: bool = False,
    n_samples: Optional[int] = None,
    seed: Optional[int] = None,
    raw_transform: Optional[Callable] = None,
    label_transform: Optional[Callable] = None,
    transform: Optional[Callable] = None,
    **loader_kwargs,
) -> DataLoader:
    """Get the HEST-1k dataloader for nuclei segmentation and cell-type classification.

    Returns batches of raw (B, 3, H, W) float32 in [0, 1] and labels (B, H, W) int32.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: Not used for cropping (patches are already 224x224); kept for API consistency.
        organs: List of organ types to include. Uses all available organs if None.
            Use PANNUKE_ORGANS for the 10-organ subset from arXiv 2604.23481.
        label_choice: Which label type to return. One of 'instances', 'xenium_instances', 'semantic'.
        download: Whether to download the data if it is not present.
        n_samples: Number of patches per epoch. Uses all patches if None.
        seed: Random seed for reproducible patch sampling when n_samples is set.
        raw_transform: Transform applied to the raw image tensor.
        label_transform: Transform applied to the label tensor.
        transform: Joint transform applied to both raw and label.
        loader_kwargs: Additional keyword arguments for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    dataset = get_hest_dataset(
        path, patch_shape, organs, label_choice, download, n_samples, seed, raw_transform, label_transform, transform
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
