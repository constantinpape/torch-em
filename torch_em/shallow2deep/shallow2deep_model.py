import os
import pickle
import torch
from torch_em.util import get_trainer, import_bioimageio_model
from .prepare_shallow2deep import _get_filters, _apply_filters


class RFWithFilters:
    def __init__(self, rf_path, ndim, filter_config):
        with open(rf_path, "rb") as f:
            self.rf = pickle.load(f)
        self.filters_and_sigmas = _get_filters(ndim, filter_config)

    def __call__(self, x):
        features = _apply_filters(x, self.filters_and_sigmas)
        assert features.shape[1] == self.rf.n_features_in_, f"{features.shape[1]}, {self.rf.n_features_in_}"
        out = self.rf.predict_proba(features)[:, 1].reshape(x.shape).astype("float32")
        return out


# TODO make sure this is picklable
class Shallow2DeepModel:

    @staticmethod
    def load_model(checkpoint, device):
        try:
            model = get_trainer(checkpoint, device=device).model
            return model
        except Exception as e:
            print("Could not load torch_em checkpoint from", checkpoint, "due to exception:", e)
            print("Trying to load as bioimageio model instead")
        model = import_bioimageio_model(checkpoint, device=device)[0]
        model.eval()
        return model

    @staticmethod
    def load_rf(rf_config):
        if len(rf_config) == 3:  # random forest path and feature config
            rf_path, ndim, filter_config = rf_config
            assert os.path.exists(rf_path)
            return RFWithFilters(rf_path, ndim, filter_config)
        elif os.path.exists(rf_config) and os.path.splitext(rf_config)[1] == ".ilp":
            # from ilastik project TODO
            pass
        else:
            raise ValueError(f"Invalid rf config: {rf_config}")

    def __init__(self, checkpoint, rf_config, device):
        self.model = self.load_model(checkpoint, device)
        self.rf_predicter = self.load_rf(rf_config)
        self.device = device

    def __call__(self, x):
        out = self.rf_predicter(x[0, 0].numpy())
        out = torch.from_numpy(out[None, None]).to(self.device)
        out = self.model(out)
        return out
