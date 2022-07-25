import os
import pickle
from glob import glob


def get_mws_config(offsets, config=None):
    mws_config = {"offsets": offsets}
    if config is None:
        config = {"mws": mws_config}
    else:
        assert isinstance(config, dict)
        config["mws"] = mws_config
    return config


def get_shallow2deep_config(rf_path, config=None):
    if os.path.isdir(rf_path):
        rf_path = glob(os.path.join(rf_path, "*.pkl"))[0]
    assert os.path.exists(rf_path), rf_path
    with open(rf_path, "rb") as f:
        rf = pickle.load(f)
    shallow2deep_config = {
        "ndim": rf.feature_ndim,
        "features": rf.feature_config,
    }
    if config is None:
        config = {"shallow2deep": shallow2deep_config}
    else:
        assert isinstance(config, dict)
        config["shallow2deep"] = shallow2deep_config
    return config
