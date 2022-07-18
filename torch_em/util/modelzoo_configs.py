

def get_mws_config(offsets, config=None):
    mws_config = {"offsets": offsets}
    if config is None:
        config = {"mws": mws_config}
    else:
        assert isinstance(config, dict)
        config["mws"] = mws_config
    return config


def get_shallow2deep_config(config=None):
    # TODO
    shallow2deep_config = {}
    if config is None:
        config = {"shallow2deep": shallow2deep_config}
    else:
        assert isinstance(config, dict)
        config["shallow2deep"] = shallow2deep_config
    return config
