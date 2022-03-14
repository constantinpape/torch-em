import bioimageio.core
import h5py
import napari
from bioimageio.core.prediction import predict_with_tiling
from xarray import DataArray


def apply_s2d_3d():
    # path to the raw data
    raw_path = "/g/emcf/pape/jil/training/pseudo_labels/rf/vanilla/block-0.h5"
    # path to the ilastik predictions
    pred_path = "/g/emcf/pape/jil/training/pseudo_labels/rf/vanilla/block-0.h5"

    with h5py.File(pred_path, "r") as f:
        # ilastik_pred = f["exported_data"][:]
        ilastik_pred = f["pseudo-labels"][:]
    print(ilastik_pred.shape)

    model_path = "/g/emcf/pape/jil/training/networks/cremi-v2/cremi-v2.zip"
    model = bioimageio.core.load_resource_description(model_path)

    with bioimageio.core.create_prediction_pipeline(bioimageio_model=model) as pp:
        input_ = DataArray(ilastik_pred[None, None], dims=tuple("bczyx"))
        pred = predict_with_tiling(pp, input_, tiling=True, verbose=True)[0].squeeze()

    with h5py.File(pred_path, "r") as f:
        raw = f["raw"][:]

    v = napari.Viewer()
    v.add_image(raw)
    v.add_image(ilastik_pred)
    v.add_image(pred)
    napari.run()


if __name__ == "__main__":
    apply_s2d_3d()
