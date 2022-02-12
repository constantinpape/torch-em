import os
import pickle
import torch
from torch_em.util import get_trainer, import_bioimageio_model
from .prepare_shallow2deep import _get_filters, _apply_filters

# optional imports only needed for using ilastik api for the predictio
try:
    import lazyflow
    from ilastik.experimental.api import from_project_file
    # set the number of threads used by ilastik to 0.
    # otherwise it does not work inside of the torch loader (and we want to limit number of threads anyways)
    # see https://github.com/ilastik/ilastik/issues/2517
    # lazyflow.request.Request.reset_thread_pool(0)
    # added this to the constructors via boolean flag
except ImportError:
    from_project_file = None
try:
    from xarray import DataArray
except ImportError:
    DataArray = None


class RFWithFilters:
    def __init__(self, rf_path, ndim, filter_config, output_channel):
        with open(rf_path, "rb") as f:
            self.rf = pickle.load(f)
        self.filters_and_sigmas = _get_filters(ndim, filter_config)
        self.output_channel = output_channel

    def __call__(self, x):
        features = _apply_filters(x, self.filters_and_sigmas)
        assert features.shape[1] == self.rf.n_features_in_, f"{features.shape[1]}, {self.rf.n_features_in_}"
        out = self.rf.predict_proba(features)[:, self.output_channel].reshape(x.shape).astype("float32")
        return out


# TODO need installation that does not downgrade numpy; talk to Dominik about this
# currently ilastik-api deps are installed via:
# conda install --strict-channel-priority -c ilastik-forge/label/freepy -c conda-forge ilastik-core
# print hint on how to install it once this is more stable
class IlastikPredicter:
    def __init__(self, ilp_path, ndim, ilastik_multi_thread, output_channel=None):
        assert from_project_file is not None
        assert DataArray is not None
        assert ndim in (2, 3)
        if not ilastik_multi_thread:
            lazyflow.request.Request.reset_thread_pool(0)
        self.ilp = from_project_file(ilp_path)
        self.dims = ("y", "x") if ndim == 2 else ("z", "y", "x")
        self.output_channel = output_channel

    def __call__(self, x):
        assert x.ndim == len(self.dims), f"{x.ndim}, {self.dims}"
        try:
            out = self.ilp.predict(DataArray(x, dims=self.dims)).values
        except ValueError as e:
            # this is a bit of a dirty hack for projects that are trained to classify in 2d, but with 3d data
            # and thus need a singleton z axis. It would be better to ask this of the ilastik classifier, see
            # https://github.com/ilastik/ilastik/issues/2530
            if x.ndim == 2:
                x = x[None]
                dims = ("z",) + self.dims
                out = self.ilp.predict(DataArray(x, dims=dims)).values
                assert out.shape[0] == 1
                # get rid of the singleton z-axis
                out = out[0]
            else:
                raise e
        if self.output_channel is not None:
            out = out[..., self.output_channel]
        return out


class Shallow2DeepModel:

    @staticmethod
    def load_model(checkpoint, device):
        try:
            model = get_trainer(checkpoint, device=device).model
            model.eval()
            return model
        except Exception as e:
            print("Could not load torch_em checkpoint from", checkpoint, "due to exception:", e)
            print("Trying to load as bioimageio model instead")
        model = import_bioimageio_model(checkpoint, device=device)[0]
        model.eval()
        return model

    @staticmethod
    def load_rf(rf_config, rf_channel, ilastik_multi_thread):
        if len(rf_config) == 3:  # random forest path and feature config
            rf_path, ndim, filter_config = rf_config
            assert os.path.exists(rf_path)
            return RFWithFilters(rf_path, ndim, filter_config, rf_channel)
        elif len(rf_config) == 2:  # ilastik project and dimensionality
            ilp_path, ndim = rf_config
            return IlastikPredicter(ilp_path, ndim, ilastik_multi_thread, rf_channel)
        else:
            raise ValueError(f"Invalid rf config: {rf_config}")

    def __init__(self, checkpoint, rf_config, device, rf_channel=1, ilastik_multi_thread=False):
        self.model = self.load_model(checkpoint, device)
        self.rf_predicter = self.load_rf(rf_config, rf_channel, ilastik_multi_thread)
        self.device = device

        self.checkpoint = checkpoint
        self.rf_config = rf_config
        self.device = device

    def __call__(self, x):
        # TODO support batch axis and multiple input channels
        out = self.rf_predicter(x[0, 0].cpu().detach().numpy())
        out = torch.from_numpy(out[None, None]).to(self.device)
        out = self.model(out)
        return out

    # need to overwrite pickle to support the rf / ilastik predicter
    def __getstate__(self):
        state = self.__dict__.copy()
        del state["rf_predicter"]
        return state

    def __setstate__(self, state):
        state["rf_predicter"] = self.load_rf(state["rf_config"])
        self.__dict__.update(state)
