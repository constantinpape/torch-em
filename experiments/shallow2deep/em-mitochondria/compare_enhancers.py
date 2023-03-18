import h5py
import numpy as np
from tqdm import trange


def predict_enhancer(enhancer_path, input_):
    tmp_path = "./bio-models/v4/predictions_uro_cell/tmp.h5"
    with h5py.File(tmp_path, "a") as f:
        if "pred" in f:
            return f["pred"][:]

        import bioimageio.core
        from xarray import DataArray
        model = bioimageio.core.load_resource_description(enhancer_path)
        pred = np.zeros((2,) + input_.shape)
        with bioimageio.core.create_prediction_pipeline(model) as pp:
            for z in trange(input_.shape[0]):
                inp = DataArray(input_[z][None, None], dims=tuple("bcyx"))
                predz = pp(inp)[0][0].values
                pred[:, z] = predz

        f.create_dataset("pred", data=pred, compression="gzip")

        return pred


def main():
    pred_path = "./bio-models/v4/predictions_uro_cell/fib1-4-3-0.h5"
    with h5py.File(pred_path, "r") as f:
        enh = f["/2d-vanilla-many-labels"][:]
        rf = f["many-labels"][:]

    old_path = "./bio-models/v2/EnhancerMitochondriaEM2D/EnhancerMitochondriaEM2D.zip"
    old = predict_enhancer(old_path, rf)

    raw_path = "/g/kreshuk/pape/Work/data/uro_cell/fib1-4-3-0.h5"
    with h5py.File(raw_path, "r") as f:
        raw = f["raw"][:]

    import napari
    v = napari.Viewer()
    v.add_image(raw)
    v.add_image(rf)
    v.add_image(enh)
    v.add_image(old)
    napari.run()


if __name__ == "__main__":
    main()
