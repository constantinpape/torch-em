import os
import pickle

import h5py
import torch
import torch_em
import torch_em.shallow2deep as shallow2deep
from torch_em.shallow2deep.prepare_shallow2deep import _get_filters, _apply_filters
from torch_em.util.util import get_trainer
from tqdm import trange

TEST_OUT = "./test_data"


def require_rf(path):
    rf_path = os.path.join(TEST_OUT, "rf_0.pkl")
    if os.path.exists(rf_path):
        return rf_path
    raw_trafo = torch_em.transform.raw.normalize
    label_trafo = shallow2deep.BoundaryTransform(ndim=2)
    shallow2deep.prepare_shallow2deep(path, "volumes/raw", path, "volumes/labels/neuron_ids",
                                      patch_shape_min=(1, 1000, 1000), patch_shape_max=(1, 1024, 1024), n_forests=1,
                                      n_threads=1, output_folder=TEST_OUT, raw_transform=raw_trafo,
                                      label_transform=label_trafo, is_seg_dataset=True, ndim=2)
    return rf_path


def _predict_rf(path, rf_path):
    out_path = os.path.join(TEST_OUT, "data.h5")
    with h5py.File(out_path, "a") as f:
        if "rf_pred" in f:
            return out_path
    print("Run prediction with rf...")
    filters_and_sigmas = _get_filters(ndim=2, filters_and_sigmas=None)
    with h5py.File(path, "r") as f:
        raw = f["volumes/raw"][:]
    with open(rf_path, "rb") as f:
        rf = pickle.load(f)

    with h5py.File(out_path, "a") as f:
        ds_out = f.create_dataset("rf_pred", shape=raw.shape, dtype="float32", chunks=(1, 512, 512))
        for z in trange(raw.shape[0], desc="Predict rf"):
            inp = raw[z].astype("float") / raw[z].max()
            feats = _apply_filters(inp, filters_and_sigmas)
            pred = rf.predict_proba(feats)[:, 1].reshape(inp.shape)
            ds_out[z] = pred
    return out_path


def _predict_enhancer(path):
    with h5py.File(path, "r") as f:
        if "enhancer_pred" in f:
            return
    with torch.no_grad():
        model = get_trainer("./checkpoints/isbi2d").model
        model.eval()
        model.to("cpu")
        with h5py.File(path, "a") as f:
            assert "rf_pred" in f
            ds_rf = f["rf_pred"]
            ds_out = f.require_dataset("enhancer_pred", shape=ds_rf.shape, dtype="float32", chunks=(1, 512, 512))
            for z in trange(ds_rf.shape[0], desc="Predict enhancer"):
                inp = ds_rf[z][:1024, :1024]
                inp = torch.from_numpy(inp[None, None])
                pred = model(inp)
                ds_out[z, :1024, :1024] = pred


def predict_s2d(path, rf_path):
    test_path = _predict_rf(path, rf_path)
    _predict_enhancer(test_path)
    return test_path


def check_prediction(path, test_path):
    with h5py.File(path, "r") as f:
        raw = f["volumes/raw"][:]
    with h5py.File(test_path, "r") as f:
        rf = f["rf_pred"][:]
        enhancer = f["enhancer_pred"][:]

    import napari
    v = napari.Viewer()
    v.add_image(raw)
    v.add_image(rf)
    v.add_image(enhancer)
    napari.run()


def main():
    os.makedirs(TEST_OUT, exist_ok=True)
    # path = "/scratch/pape/cremi/sampleA.h5"
    path = "/home/pape/Work/data/cremi/sample_A_20160501.hdf"
    rf_path = require_rf(path)
    test_path = predict_s2d(path, rf_path)
    check_prediction(path, test_path)


if __name__ == "__main__":
    main()
