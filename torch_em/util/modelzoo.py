import argparse
import functools
import json
import os
import pickle
import subprocess
from glob import glob
from pathlib import Path
from shutil import copyfile
from warnings import warn
from zipfile import ZipFile

import imageio
import numpy as np
import torch
import torch_em

import bioimageio.core as core
import bioimageio.core.build_spec as build_spec
from bioimageio.core.prediction_pipeline._model_adapters._pytorch_model_adapter import PytorchModelAdapter
import bioimageio.core.weight_converter.torch as weight_converter
from bioimageio.spec.shared import yaml

from elf.io import open_file
from .util import get_trainer, get_normalizer


#
# general purpose functionality
#


def normalize_with_batch(data, normalizer):
    if normalizer is None:
        return data
    normalized = np.concatenate(
        [normalizer(da)[None] for da in data],
        axis=0
    )
    return normalized


#
# utility functions for model export
#


def get_default_citations(model=None, model_output=None):
    citations = [
        {"text": "training library", "doi": "https://doi.org/10.5281/zenodo.5108853"}
    ]

    # try to derive the correct network citation from the model class
    if model is not None:
        if isinstance(model, str):
            model_name = model
        else:
            model_name = str(model.__class__.__name__)

        if model_name.lower() in ("unet2d", "unet_2d", "unet"):
            citations.append(
                {"text": "architecture", "doi": "https://doi.org/10.1007/978-3-319-24574-4_28"}
            )
        elif model_name.lower() in ("unet3d", "unet_3d", "anisotropicunet"):
            citations.append(
                {"text": "architecture", "doi": "https://doi.org/10.1007/978-3-319-46723-8_49"}
            )
        else:
            warn("No citation for architecture {model_name} found.")

    # try to derive the correct segmentation algo citation from the model output type
    if model_output is not None:
        msg = f"No segmentation algorithm for output {model_output} known. 'affinities' and 'boundaries' are supported."
        if model_output == "affinities":
            citations.append(
                {"text": "segmentation algorithm", "doi": "https://doi.org/10.1109/TPAMI.2020.2980827"}
            )
        elif model_output == "boundaries":
            citations.append(
                {"text": "segmentation algorithm", "doi": "https://doi.org/10.1038/nmeth.4151"}
            )
        else:
            warn(msg)

    return citations


def _get_model(trainer, postprocessing):
    model = trainer.model
    model.eval()
    model_kwargs = model.init_kwargs
    # clear the kwargs of non builtins
    # TODO warn if we strip any non-standard arguments
    model_kwargs = {k: v for k, v in model_kwargs.items() if not isinstance(v, type)}

    # set the in-model postprocessing if given
    if postprocessing is not None:
        assert "postprocessing" in model_kwargs
        model_kwargs["postprocessing"] = postprocessing
        state = model.state_dict()
        model = model.__class__(**model_kwargs)
        model.load_state_dict(state)
        model.eval()

    return model, model_kwargs


def _pad(input_data, trainer):
    try:
        ndim = trainer.train_loader.dataset.ndim
    except AttributeError:
        ndim = trainer.train_loader.dataset.datasets[0].ndim
    target_dims = ndim + 2
    for _ in range(target_dims - input_data.ndim):
        input_data = np.expand_dims(input_data, axis=0)
    return input_data


def _write_depedencies(export_folder, dependencies):
    dep_path = os.path.join(export_folder, "environment.yaml")
    if dependencies is None:
        ver = torch.__version__
        major, minor = list(map(int, ver.split(".")[:2]))
        assert major in (1, 2)
        if major == 2:
            warn("Modelzoo functionality is not fully tested for PyTorch 2")
        # the torch zip layout changed for a few versions:
        torch_min_version = "1.0"
        if minor > 6 and minor < 10:
            torch_min_version = "1.6"
        else:
            torch_min_version = "1.10"
        torch_min_version = "1.6" if minor >= 6 else "1.0"
        dependencies = {
            "channels": ["pytorch", "conda-forge"],
            "name": "torch-em-deploy",
            "dependencies": [f"pytorch>={torch_min_version}"]
        }
        with open(dep_path, "w") as f:
            yaml.dump(dependencies, f)
    else:
        assert os.path.exists(dependencies)
        dep = yaml.load(dependencies)
        assert "channels" in dep
        assert "name" in dep
        assert "dependencies" in dep
        copyfile(dependencies, dep_path)


def _write_data(input_data, model, trainer, export_folder):
    # if input_data is None:
    #     gen = SampleGenerator(trainer, 1, False, 1)
    #     input_data = next(gen)
    if isinstance(input_data, np.ndarray):
        input_data = [input_data]

    # normalize the input data if we have a normalization function
    normalizer = get_normalizer(trainer)

    # pad to 4d/5d and normalize the input data
    # NOTE we have to save the padded data, but without normalization
    test_inputs = [_pad(input_, trainer) for input_ in input_data]
    normalized = [normalize_with_batch(input_, normalizer) for input_ in test_inputs]

    # run prediction
    with torch.no_grad():
        test_tensors = [torch.from_numpy(norm).to(trainer.device) for norm in normalized]
        test_outputs = model(*test_tensors)
        if torch.is_tensor(test_outputs):
            test_outputs = [test_outputs]
        test_outputs = [out.cpu().numpy() for out in test_outputs]

    # save the input / output
    test_in_paths, test_out_paths = [], []
    for i, input_ in enumerate(test_inputs):
        test_in_path = os.path.join(export_folder, f"test_input_{i}.npy")
        np.save(test_in_path, input_)
        test_in_paths.append(test_in_path)
    for i, out in enumerate(test_outputs):
        test_out_path = os.path.join(export_folder, f"test_output_{i}.npy")
        np.save(test_out_path, out)
        test_out_paths.append(test_out_path)
    return test_in_paths, test_out_paths


def _write_source(model, export_folder):
    # copy the model source file if it"s a torch_em model
    # (for now only u-net). otherwise just put the full python class
    module = str(model.__class__.__module__)
    cls_name = str(model.__class__.__name__)
    if module == "torch_em.model.unet":
        source_path = os.path.join(
            os.path.split(__file__)[0],
            "../model/unet.py"
        )
        source_target_path = os.path.join(export_folder, "unet.py")
        copyfile(source_path, source_target_path)
        source = f"./unet.py::{cls_name}"
    else:
        source = f"{module}.{cls_name}"
    return source


def _get_kwargs(trainer, name, description,
                authors, tags,
                license, documentation,
                git_repo, cite,
                maintainers,
                export_folder, input_optional_parameters):
    if input_optional_parameters:
        print("Enter values for the optional parameters.")
        print("If the default value in [] is satisfactory, press enter without additional input.")
        print("Please enter lists using json syntax.")

    def _get_kwarg(kwarg_name, val, default, is_list=False, fname=None):
        # if we don"t have a value, we either ask user for input (offering the default)
        # or just use the default if input_optional_parameters is False
        if val is None and input_optional_parameters:
            default_val = default()
            choice = input(f"{kwarg_name} [{default_val}]: ")
            val = choice if choice else default_val
        elif val is None:
            val = default()

        if fname is not None:
            save_path = os.path.join(export_folder, fname)
            with open(save_path, "w") as f:
                f.write(val)
            return f"./{fname}"

        if is_list and isinstance(val, str):
            val = val.replace(""", """)  # enable single quotes
            val = json.loads(val)
        if is_list:
            assert isinstance(val, (list, tuple)), type(val)
        return val

    def _default_authors():
        # first try to derive the author name from git
        try:
            call_res = subprocess.run(["git", "config", "user.name"], capture_output=True)
            author = call_res.stdout.decode("utf8").rstrip("\n")
            author = author if author else None  # in case there was no error, but output is empty
        except Exception:
            author = None

        # otherwise use the username
        if author is None:
            author = os.uname()[1]

        return [{"name": author}]

    def _default_repo():
        return None
        try:
            call_res = subprocess.run(["git", "remote", "-v"], capture_output=True)
            repo = call_res.stdout.decode("utf8").split("\n")[0].split()[1]
            repo = repo if repo else None
        except Exception:
            repo = None
        return repo

    def _default_maintainers():
        # first try to derive the maintainer name from git
        try:
            call_res = subprocess.run(["git", "config", "user.name"], capture_output=True)
            maintainer = call_res.stdout.decode("utf8").rstrip("\n")
            maintainer = maintainer if maintainer else None  # in case there was no error, but output is empty
        except Exception:
            maintainer = None

        # otherwise use the username
        if maintainer is None:
            maintainer = os.uname()[1]

        return [{"github_user": maintainer}]

    # TODO derive better default values:
    # - description: derive something from trainer.ndim, trainer.loss, trainer.model, ...
    # - tags: derive something from trainer.ndim, trainer.loss, trainer.model, ...
    # - documentation: derive something from trainer.ndim, trainer.loss, trainer.model, ...
    kwargs = {
        "name": _get_kwarg("name", name, lambda: trainer.name),
        "description": _get_kwarg("description", description, lambda: trainer.name),
        "authors": _get_kwarg("authors", authors, _default_authors, is_list=True),
        "tags": _get_kwarg("tags", tags, lambda: [trainer.name], is_list=True),
        "license": _get_kwarg("license", license, lambda: "MIT"),
        "documentation": _get_kwarg("documentation", documentation, lambda: trainer.name,
                                    fname="documentation.md"),
        "git_repo": _get_kwarg("git_repo", git_repo, _default_repo),
        "cite": _get_kwarg("cite", cite, get_default_citations),
        "maintainers": _get_kwarg("maintainers", maintainers, _default_maintainers, is_list=True),
    }

    return kwargs


def _write_weights(model, export_folder):
    weights = model.state_dict()
    weight_name = "weights.pt"
    weight_path = os.path.join(export_folder, weight_name)
    torch.save(weights, weight_path)
    return f"./{weight_name}"


def _get_preprocessing(trainer):
    try:
        ndim = trainer.train_loader.dataset.ndim
    except AttributeError:
        ndim = trainer.train_loader.dataset.datasets[0].ndim
    normalizer = get_normalizer(trainer)

    if isinstance(normalizer, functools.partial):
        kwargs = normalizer.keywords
        normalizer = normalizer.func
    else:
        kwargs = {}

    def _get_axes(axis):
        if axis is None:
            axes = "cyx" if ndim == 2 else "czyx"
        else:
            axes_labels = ["c", "y", "x"] if ndim == 2 else ["c", "z", "y", "x"]
            axes = "".join(axes_labels[i] for i in axis)
        return axes

    name = f"{normalizer.__module__}.{normalizer.__name__}"
    if name == "torch_em.transform.raw.normalize":

        min_, max_ = kwargs.get("minval", None), kwargs.get("maxval", None)
        axes = _get_axes(kwargs.get("axis", None))
        assert (min_ is None) == (max_ is None)

        if min_ is None:
            preprocessing = [{
                "name": "scale_range",
                "kwargs": {"mode": "per_sample", "axes": axes,
                           "min_percentile": 0.0, "max_percentile": 100.0}
            }]
        else:
            preprocessing = [{
                "name": "scale_linear",
                "kwargs": {"gain": 1. / max_, "offset": -min_, "axes": axes}
            }]

    elif name == "torch_em.transform.raw.standardize":

        mean, std = kwargs.get("mean", None), kwargs.get("std", None)
        mode = "per_sample" if mean is None else "fixed"
        axes = _get_axes(kwargs.get("axis", None))
        preprocessing = [{
            "name": "zero_mean_unit_variance",
            "kwargs": {"mode": mode, "axes": axes}
        }]
        if mean is not None:
            preprocessing[0]["kwargs"]["mean"] = mean
        if std is not None:
            preprocessing[0]["kwargs"]["std"] = std

    elif name == "torch_em.transform.normalize_percentile":

        lower, upper = kwargs.get("lower", 1.0), kwargs.get("upper", 99.0)
        axes = _get_axes(kwargs.get("axis", None))
        preprocessing = [{
            "name": "scale_range",
            "kwargs": {"mode": "per_sample", "axes": axes,
                       "min_percentile": lower, "max_percentile": upper}
        }]

    else:
        warn("Could not parse the normalization function, 'preprocessing' field will be empty.")
        return None

    return [preprocessing]


def _get_tensor_kwargs(model, model_kwargs, input_tensors, output_tensors, min_shape, halo):

    def get_ax(tensor):
        ndim = np.load(tensor).ndim
        assert ndim in (4, 5)
        return "bcyx" if ndim == 4 else "bczyx"

    tensor_kwargs = {
        "input_axes": [get_ax(tensor) for tensor in input_tensors],
        "output_axes": [get_ax(tensor) for tensor in output_tensors]
    }
    notebook_link = None
    module = str(model.__class__.__module__)
    name = str(model.__class__.__name__)

    # can derive tensor kwargs only for known torch_em models (only unet for now)
    if module == "torch_em.model.unet":
        assert len(input_tensors) == len(output_tensors) == 1
        inc, outc = model_kwargs["in_channels"], model_kwargs["out_channels"]

        postprocessing = model_kwargs.get("postprocessing", None)
        if isinstance(postprocessing, str) and postprocessing.startswith("affinities_to_boundaries"):
            outc = 1
        elif isinstance(postprocessing, str) and postprocessing.startswith("affinities_with_foreground_to_boundaries"):
            outc = 2
        elif postprocessing is not None:
            warn(f"The model has the post-processing {postprocessing} which cannot be interpreted")

        if name == "UNet2d":
            depth = model_kwargs["depth"]
            step = [0, 0] + [2 ** depth] * 2
            if min_shape is None:
                min_shape = [1, inc] + [2 ** (depth + 1)] * 2
            else:
                assert len(min_shape) == 2
                min_shape = [1, inc] + list(min_shape)
            notebook_link = "ilastik/torch-em-2d-unet-notebook"
        elif name == "UNet3d":
            depth = model_kwargs["depth"]
            step = [0, 0] + [2 ** depth] * 3
            if min_shape is None:
                min_shape = [1, inc] + [2 ** (depth + 1)] * 3
            else:
                assert len(min_shape) == 3
                min_shape = [1, inc] + list(min_shape)
            notebook_link = "ilastik/torch-em-3d-unet-notebook"
        elif name == "AnisotropicUNet":
            scale_factors = model_kwargs["scale_factors"]
            scale_prod = [
                int(np.prod([scale_factors[i][d] for i in range(len(scale_factors))]))
                for d in range(3)
            ]
            assert len(scale_prod) == 3
            step = [0, 0] + scale_prod
            if min_shape is None:
                min_shape = [1, inc] + [2 * sp for sp in scale_prod]
            else:
                min_shape = [1, inc] + list(min_shape)
            assert len(min_shape) == len(step), f"{len(min_shape), len(step)}"
            notebook_link = "ilastik/torch-em-3d-unet-notebook"
        else:
            raise RuntimeError(f"Cannot derive tensor parameters for {module}.{name}")

        if halo is None:   # default halo = step // 2
            halo = [st // 2 for st in step]
        else:  # make sure the passed halo has the same length as step, by padding with zeros
            halo = [0] * (len(step) - len(halo)) + halo
        assert len(halo) == len(step), f"{len(halo)}, {len(step)}"

        ref = "input0"
        if inc == outc:
            scale = [1] * len(step)
            offset = [0] * len(step)
        else:
            scale = [1, float(outc) / inc] + ([1] * (len(step) - 2))
            offset = [0, 0] + ([0] * (len(step) - 2))
        tensor_kwargs.update({
            "input_step": [step],
            "input_min_shape": [min_shape],
            "output_reference": [ref],
            "output_scale": [scale],
            "output_offset": [offset],
            "halo": [halo]
        })
    return tensor_kwargs, notebook_link


def _validate_model(spec_path):
    if not os.path.exists(spec_path):
        return False

    model, normalizer, model_spec = import_bioimageio_model(spec_path, return_spec=True)
    inputs = [normalize_with_batch(np.load(test_in), normalizer) for test_in in model_spec.test_inputs]
    expected = [np.load(test_out) for test_out in model_spec.test_outputs]

    with torch.no_grad():
        inputs = [torch.from_numpy(input_) for input_ in inputs]
        outputs = model(*inputs)
        if torch.is_tensor(outputs):
            outputs = [outputs]
        outputs = [out.numpy() for out in outputs]

    for out, exp in zip(outputs, expected):
        if not np.allclose(out, exp):
            return False
    return True


def _extract_from_zip(zip_path, out_path, name):
    with ZipFile(zip_path) as z:
        with open(out_path, "w") as f:
            f.write(z.read(name).decode("utf-8"))


#
# model export functionality
#

def _get_input_data(trainer):
    loader = trainer.val_loader
    x = next(iter(loader))[0].numpy()
    return x


# TODO config: training details derived from loss and optimizer, custom params, e.g. offsets for mws
def export_bioimageio_model(checkpoint, export_folder, input_data=None,
                            dependencies=None, name=None,
                            description=None, authors=None,
                            tags=None, license=None,
                            documentation=None, covers=None,
                            git_repo=None, cite=None,
                            input_optional_parameters=True,
                            model_postprocessing=None,
                            for_deepimagej=False, links=None,
                            maintainers=None, min_shape=None, halo=None,
                            checkpoint_name="best",
                            training_data=None, config={}):
    """
    """
    # load trainer and model
    trainer = get_trainer(checkpoint, name=checkpoint_name, device="cpu")
    model, model_kwargs = _get_model(trainer, model_postprocessing)

    if input_data is None:
        input_data = _get_input_data(trainer)

    # create the weights
    os.makedirs(export_folder, exist_ok=True)
    weight_path = _write_weights(model, export_folder)

    # create the test input/output file and derive the tensor kwargs from the model and its kwargs
    test_in_paths, test_out_paths = _write_data(input_data, model, trainer, export_folder)
    tensor_kwargs, notebook_link = _get_tensor_kwargs(
        model, model_kwargs, test_in_paths, test_out_paths, min_shape, halo
    )

    # create the model source file
    source = _write_source(model, export_folder)

    # create dependency file
    _write_depedencies(export_folder, dependencies)

    # get the additional kwargs
    kwargs = _get_kwargs(trainer, name, description,
                         authors, tags,
                         license, documentation,
                         git_repo, cite,
                         maintainers,
                         export_folder, input_optional_parameters)
    kwargs.update(tensor_kwargs)
    preprocessing = _get_preprocessing(trainer)

    # the apps to link with this model, by default ilastik
    if links is None:
        links = []
    links.append("ilastik/ilastik")
    # add the notebook link, if available
    if notebook_link is not None:
        links.append(notebook_link)
    kwargs.update({"links": links, "config": config})

    zip_path = os.path.join(export_folder, f"{name}.zip")
    # change the working directory to the export_folder to avoid issues with relative paths
    cwd = os.getcwd()
    os.chdir(export_folder)

    try:
        build_spec.build_model(
            weight_uri=weight_path,
            weight_type="pytorch_state_dict",
            test_inputs=[f"./{os.path.split(test_in)[1]}" for test_in in test_in_paths],
            test_outputs=[f"./{os.path.split(test_out)[1]}" for test_out in test_out_paths],
            root=".",
            output_path=f"{name}.zip",
            dependencies="environment.yaml",
            preprocessing=preprocessing,
            architecture=source,
            model_kwargs=model_kwargs,
            add_deepimagej_config=for_deepimagej,
            training_data=training_data,
            **kwargs
        )
    except Exception as e:
        raise e
    finally:
        os.chdir(cwd)

    # load and validate the model
    rdf_path = os.path.join(export_folder, "rdf.yaml")
    _extract_from_zip(zip_path, rdf_path, "rdf.yaml")
    val_success = _validate_model(rdf_path)

    if val_success:
        print(f"The model was successfully exported to '{export_folder}'.")
    else:
        warn(f"Validation of the bioimageio model exported to '{export_folder}' has failed. " +
             "You can use this model, but it will probably yield incorrect results.")
    return val_success


# TODO support bounding boxes
def _load_data(path, key):
    if key is None:
        ext = os.path.splitext(path)[-1]
        if ext == ".npy":
            return np.load(path)
        else:
            return imageio.imread(path)
    else:
        return open_file(path, mode="r")[key][:]


def main():
    import argparse
    parser = argparse.ArgumentParser(
        "Export model trained with torch_em to the BioImage.IO model format."
        "The exported model can be run in any tool supporting BioImage.IO."
        "For details check out https://bioimage.io/#/."
    )
    parser.add_argument("-p", "--path", required=True,
                        help="Path to the model checkpoint to export to the BioImage.IO model format.")
    parser.add_argument("-d", "--data", required=True,
                        help="Path to the test data to use for creating the exported model.")
    parser.add_argument("-f", "--export_folder", required=True,
                        help="Where to save the exported model. The exported model is stored as a zip in the folder.")
    parser.add_argument("-k", "--key",
                        help="The key for the test data. Required for container data formats like hdf5 or zarr.")
    parser.add_argument("-n", "--name", help="The name of the exported model.")

    args = parser.parse_args()
    name = os.path.basename(args.path) if args.name is None else args.name
    export_bioimageio_model(args.path, args.export_folder, _load_data(args.data, args.key), name=name)


#
# model import functionality
#

def _load_model(model_spec, device):
    model = PytorchModelAdapter.get_nn_instance(model_spec)
    weights = model_spec.weights["pytorch_state_dict"]
    state = torch.load(weights.source, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def _load_normalizer(model_spec):
    inputs = model_spec.inputs[0]
    preprocessing = inputs.preprocessing
    if len(preprocessing) == 0:
        return None

    ndim = len(inputs.axes) - 2
    shape = inputs.shape
    if hasattr(shape, "min"):
        shape = shape.min

    conf = preprocessing[0]
    name = conf.name
    spec_kwargs = conf.kwargs

    def _get_axis(axes):
        label_to_id = {"c": 0, "z": 1, "y": 2, "x": 3} if ndim == 3 else\
            {"c": 0, "y": 1, "x": 2}
        axis = tuple(label_to_id[ax] for ax in axes)

        # is the axis full? Then we don"t need to specify it.
        if len(axis) == ndim + 1:
            return None

        # drop the channel axis if we have only a single channel
        # (because torch_em squeezes the channel axis in this case)
        if shape[1] == 1:
            axis = tuple(ax - 1 for ax in axis if ax > 0)
        return axis

    if name == "zero_mean_unit_variance":
        mode = spec_kwargs["mode"]
        if mode == "fixed":
            kwargs = {"mean": spec_kwargs["mean"], "std": spec_kwargs["std"]}
        else:
            axis = _get_axis(spec_kwargs["axes"])
            kwargs = {"axis": axis}
        normalizer = functools.partial(
            torch_em.transform.raw.standardize,
            **kwargs
        )

    elif name == "scale_linear":
        axis = _get_axis(spec_kwargs["axes"])
        min_ = -spec_kwargs["offset"]
        max_ = 1. / spec_kwargs["gain"]
        kwargs = {"axis": axis, "minval": min_, "maxval": max_}
        normalizer = functools.partial(
            torch_em.transform.raw.normalize,
            **kwargs
        )

    elif name == "scale_range":
        assert spec_kwargs["mode"] == "per_sample"  # can"t parse the other modes right now
        axis = _get_axis(spec_kwargs["axes"])
        lower, upper = spec_kwargs["min_percentile"], spec_kwargs["max_percentile"]
        if np.isclose(lower, 0.0) and np.isclose(upper, 100.0):
            normalizer = functools.partial(
                torch_em.transform.raw.normalize,
                axis=axis
            )
        else:
            kwargs = {"axis": axis, "lower": lower, "upper": upper}
            normalizer = functools.partial(
                torch_em.transform.raw.normalize_percentile,
                **kwargs
            )

    else:
        msg = f"torch_em does not support the use of the biomageio preprocessing function {name}"
        raise RuntimeError(msg)

    return normalizer


def import_bioimageio_model(spec_path, return_spec=False, device="cpu"):
    model_spec = core.load_resource_description(spec_path)

    model = _load_model(model_spec, device=device)
    normalizer = _load_normalizer(model_spec)

    if return_spec:
        return model, normalizer, model_spec
    else:
        return model, normalizer


# TODO
def import_trainer_from_bioimageio_model(spec_path):
    pass


#
# weight conversion
#


def _convert_impl(spec_path, weight_name, converter, weight_type, **kwargs):
    root = Path(os.path.split(spec_path)[0])
    if isinstance(spec_path, str):
        spec_path = Path(spec_path)
    weight_path = os.path.join(root, weight_name)

    # here, we need the model with resolved nodes
    model_spec = core.load_resource_description(spec_path)
    converter(model_spec, weight_path, **kwargs)

    # now, we need the model with raw nodes
    model_spec = core.load_raw_resource_description(spec_path)
    zip_path = os.path.join(root, f"{model_spec.name}.zip")
    model_spec = build_spec.add_weights(
        model_spec, weight_path, weight_type=weight_type, output_path=zip_path, **kwargs
    )
    rdf_path = os.path.join(root, "rdf.yaml")
    _extract_from_zip(zip_path, rdf_path, "rdf.yaml")


def convert_to_onnx(spec_path, opset_version=12):
    converter = weight_converter.convert_weights_to_onnx
    _convert_impl(spec_path, "weights.onnx", converter, "onnx", opset_version=opset_version)
    # TODO check the exported model and return exception if it fails
    return None


def convert_to_torchscript(spec_path):
    converter = weight_converter.convert_weights_to_torchscript
    weight_name = "weights-torchscript.pt"
    _convert_impl(spec_path, weight_name, converter, "torchscript")

    # check that we can actually load it again
    root = os.path.split(spec_path)[0]
    weight_path = os.path.join(root, weight_name)
    try:
        torch.jit.load(weight_path)
        return None
    except Exception as e:
        return e


def add_weight_formats(export_folder, additional_formats):
    spec_path = os.path.join(export_folder, "rdf.yaml")
    for add_format in additional_formats:

        if add_format == "onnx":
            ret = convert_to_onnx(spec_path)
        elif add_format == "torchscript":
            ret = convert_to_torchscript(spec_path)

        if ret is None:
            print("Successfully added", add_format, "weights")
        else:
            warn(f"Added {add_format} weights, but got exception {ret} when loading the weights again.")


def convert_main():
    import argparse
    parser = argparse.ArgumentParser(
        "Convert weights from native pytorch format to onnx or torchscript"
    )
    parser.add_argument("-f", "--model_folder", required=True,
                        help="")
    parser.add_argument("-w", "--weight_format", required=True,
                        help="")
    args = parser.parse_args()
    weight_format = args.weight_format
    assert weight_format in ("onnx", "torchscript")
    if weight_format == "onnx":
        convert_to_onnx(args.model_folder)
    else:
        convert_to_torchscript(args.model_folder)


#
# misc functionality
#

def export_parser_helper():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", required=True)
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-a", "--affs_to_bd", default=0, type=int)
    parser.add_argument("-f", "--additional_formats", type=str, nargs="+")
    return parser


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
