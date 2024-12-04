import argparse
import functools
import json
import os
import pickle
import subprocess
import tempfile

from glob import glob
from pathlib import Path
from warnings import warn

import imageio
import numpy as np
import torch
import torch_em

import bioimageio.core as core
import bioimageio.spec.model.v0_5 as spec
from bioimageio.core.model_adapters._pytorch_model_adapter import PytorchModelAdapter
from bioimageio.spec import save_bioimageio_package

from elf.io import open_file
from .util import get_trainer, get_normalizer


#
# General Purpose Functionality
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
# Utility Functions for Model Export.
#


def get_default_citations(model=None, model_output=None):
    citations = [
        {"text": "training library", "doi": "10.5281/zenodo.5108853"}
    ]

    # try to derive the correct network citation from the model class
    if model is not None:
        if isinstance(model, str):
            model_name = model
        else:
            model_name = str(model.__class__.__name__)

        if model_name.lower() in ("unet2d", "unet_2d", "unet"):
            citations.append(
                {"text": "architecture", "doi": "10.1007/978-3-319-24574-4_28"}
            )
        elif model_name.lower() in ("unet3d", "unet_3d", "anisotropicunet"):
            citations.append(
                {"text": "architecture", "doi": "10.1007/978-3-319-46723-8_49"}
            )
        else:
            warn("No citation for architecture {model_name} found.")

    # try to derive the correct segmentation algo citation from the model output type
    if model_output is not None:
        msg = f"No segmentation algorithm for output {model_output} known. 'affinities' and 'boundaries' are supported."
        if model_output == "affinities":
            citations.append(
                {"text": "segmentation algorithm", "doi": "10.1109/TPAMI.2020.2980827"}
            )
        elif model_output == "boundaries":
            citations.append(
                {"text": "segmentation algorithm", "doi": "10.1038/nmeth.4151"}
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
        if isinstance(trainer.train_loader.dataset, torch.utils.data.dataset.Subset):
            ndim = trainer.train_loader.dataset.dataset.ndim
        else:
            ndim = trainer.train_loader.dataset.ndim
    except AttributeError:
        ndim = trainer.train_loader.dataset.datasets[0].ndim
    target_dims = ndim + 2
    for _ in range(target_dims - input_data.ndim):
        input_data = np.expand_dims(input_data, axis=0)
    return input_data


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


def _create_weight_description(model, export_folder, model_kwargs):
    module = str(model.__class__.__module__)
    cls_name = str(model.__class__.__name__)

    if module == "torch_em.model.unet":
        source_path = os.path.join(os.path.split(__file__)[0], "../model/unet.py")
        architecture = spec.ArchitectureFromFileDescr(
            source=Path(source_path),
            callable=cls_name,
            kwargs=model_kwargs,
        )
    else:
        architecture = spec.ArchitectureFromLibraryDescr(
            import_from=module,
            callable=cls_name,
            kwargs=model_kwargs,
        )

    checkpoint_path = os.path.join(export_folder, "state_dict.pt")
    torch.save(model.state_dict(), checkpoint_path)

    weight_description = spec.WeightsDescr(
        pytorch_state_dict=spec.PytorchStateDictWeightsDescr(
            source=Path(checkpoint_path),
            architecture=architecture,
            pytorch_version=spec.Version(torch.__version__),
        )
    )
    return weight_description


def _get_kwargs(
    trainer, name, description, authors, tags, license, documentation,
    git_repo, cite, maintainers, export_folder, input_optional_parameters
):
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
            return save_path

        if is_list and isinstance(val, str):
            val = val.replace("'", '"')  # enable single quotes
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
        "documentation": _get_kwarg(
            "documentation", documentation, lambda: trainer.name, fname="documentation.md"
        ),
        "git_repo": _get_kwarg("git_repo", git_repo, _default_repo),
        "cite": _get_kwarg("cite", cite, get_default_citations),
        "maintainers": _get_kwarg("maintainers", maintainers, _default_maintainers, is_list=True),
    }

    return kwargs


def _get_preprocessing(trainer):
    try:
        if isinstance(trainer.train_loader.dataset, torch.utils.data.dataset.Subset):
            ndim = trainer.train_loader.dataset.dataset.ndim
        else:
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
        all_axes = ["channel", "y", "x"] if ndim == 2 else ["channel", "z", "y", "x"]
        if axis is None:
            axes = all_axes
        else:
            axes = [all_axes[i] for i in axes]
        return axes

    name = f"{normalizer.__module__}.{normalizer.__name__}"
    axes = _get_axes(kwargs.get("axis", None))

    if name == "torch_em.transform.raw.normalize":

        min_, max_ = kwargs.get("minval", None), kwargs.get("maxval", None)
        assert (min_ is None) == (max_ is None)

        if min_ is None:
            spec_name = "scale_range",
            spec_kwargs = {"mode": "per_sample", "axes": axes, "min_percentile": 0.0, "max_percentile": 100.0}
        else:
            spec_name = "scale_linear"
            spec_kwargs = {"gain": 1.0 / max_, "offset": -min_, "axes": axes}

    elif name == "torch_em.transform.raw.standardize":
        spec_kwargs = {"axes": axes}
        mean, std = kwargs.get("mean", None), kwargs.get("std", None)
        if (mean is None) and (std is None):
            spec_name = "zero_mean_unit_variance"
        else:
            spec_name = "fixed_zero_mean_unit_varaince"
            spec_kwargs.update({"mean": mean, "std": std})

    elif name == "torch_em.transform.raw.normalize_percentile":
        lower, upper = kwargs.get("lower", 1.0), kwargs.get("upper", 99.0)
        spec_name = "scale_range"
        spec_kwargs = {"mode": "per_sample", "axes": axes, "min_percentile": lower, "max_percentile": upper}

    else:
        warn(f"Could not parse the normalization function {name}, 'preprocessing' field will be empty.")
        return None

    name_to_cls = {
        "scale_linear": spec.ScaleLinearDescr,
        "scale_rage": spec.ScaleRangeDescr,
        "zero_mean_unit_variance": spec.ZeroMeanUnitVarianceDescr,
        "fixed_zero_mean_unit_variance": spec.FixedZeroMeanUnitVarianceDescr,
    }
    preprocessing = name_to_cls[spec_name](kwargs=spec_kwargs)

    return [preprocessing]


def _get_inout_descriptions(trainer, model, model_kwargs, input_tensors, output_tensors, min_shape, halo):

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
            step = [2 ** depth] * 2
            if min_shape is None:
                min_shape = [2 ** (depth + 1)] * 2
            else:
                assert len(min_shape) == 2
                min_shape = list(min_shape)
            notebook_link = "ilastik/torch-em-2d-unet-notebook"

        elif name == "UNet3d":
            depth = model_kwargs["depth"]
            step = [2 ** depth] * 3
            if min_shape is None:
                min_shape = [2 ** (depth + 1)] * 3
            else:
                assert len(min_shape) == 3
                min_shape = list(min_shape)
            notebook_link = "ilastik/torch-em-3d-unet-notebook"

        elif name == "AnisotropicUNet":
            scale_factors = model_kwargs["scale_factors"]
            scale_prod = [
                int(np.prod([scale_factors[i][d] for i in range(len(scale_factors))]))
                for d in range(3)
            ]
            assert len(scale_prod) == 3
            step = scale_prod
            if min_shape is None:
                min_shape = [2 * sp for sp in scale_prod]
            else:
                min_shape = list(min_shape)
            notebook_link = "ilastik/torch-em-3d-unet-notebook"

        else:
            raise RuntimeError(f"Cannot derive tensor parameters for {module}.{name}")

        if halo is None:   # default halo = step // 2
            halo = [st // 2 for st in step]
        else:  # make sure the passed halo has the same length as step, by padding with zeros
            halo = [0] * (len(step) - len(halo)) + halo
        assert len(halo) == len(step), f"{len(halo)}, {len(step)}"

        # Create the input axis description.
        input_axes = [
            spec.BatchAxis(),
            spec.ChannelAxis(channel_names=[spec.Identifier(f"channel_{i}") for i in range(inc)]),
        ]
        input_ndim = np.load(input_tensors[0]).ndim
        assert input_ndim in (4, 5)
        axis_names = "zyx" if input_ndim == 5 else "yx"
        assert len(axis_names) == len(min_shape) == len(step)
        input_axes += [
            spec.SpaceInputAxis(id=spec.AxisId(ax_name), size=spec.ParameterizedSize(min=ax_min, step=ax_step))
            for ax_name, ax_min, ax_step in zip(axis_names, min_shape, step)
        ]

        # Create the rest of the input description.
        preprocessing = _get_preprocessing(trainer)
        input_description = [spec.InputTensorDescr(
            id=spec.TensorId("image"),
            axes=input_axes,
            test_tensor=spec.FileDescr(source=Path(input_tensors[0])),
            preprocessing=preprocessing,
        )]

        # Create the output axis description.
        output_axes = [
            spec.BatchAxis(),
            spec.ChannelAxis(channel_names=[spec.Identifier(f"out_channel_{i}") for i in range(outc)]),
        ]
        output_ndim = np.load(output_tensors[0]).ndim
        assert output_ndim in (4, 5)
        axis_names = "zyx" if output_ndim == 5 else "yx"
        assert len(axis_names) == len(halo)
        output_axes += [
            spec.SpaceOutputAxisWithHalo(
                id=spec.AxisId(ax_name),
                size=spec.SizeReference(
                    tensor_id=spec.TensorId("image"), axis_id=spec.AxisId(ax_name)
                ),
                halo=halo_val,
            ) for ax_name, halo_val in zip(axis_names, halo)
        ]

        # Create the rest of the output description.
        output_description = [spec.OutputTensorDescr(
            id=spec.TensorId("prediction"),
            axes=output_axes,
            test_tensor=spec.FileDescr(source=Path(output_tensors[0]))
        )]

    else:
        raise NotImplementedError("Model export currently only works for torch_em.model.unet.")

    return input_description, output_description, notebook_link


def _validate_model(spec_path):
    if not os.path.exists(spec_path):
        return False

    model, normalizer, model_spec = import_bioimageio_model(spec_path, return_spec=True)
    root = model_spec.root

    input_paths = [os.path.join(root, ipt.test_tensor.source.path) for ipt in model_spec.inputs]
    inputs = [normalize_with_batch(np.load(ipt), normalizer) for ipt in input_paths]

    expected_paths = [os.path.join(root, opt.test_tensor.source.path) for opt in model_spec.outputs]
    expected = [np.load(opt) for opt in expected_paths]

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


#
# Model Export Functionality
#

def _get_input_data(trainer):
    loader = trainer.val_loader
    x = next(iter(loader))[0].numpy()
    return x


# TODO config: training details derived from loss and optimizer, custom params, e.g. offsets for mws
def export_bioimageio_model(
    checkpoint,
    output_path,
    input_data=None,
    name=None,
    description=None,
    authors=None,
    tags=None,
    license=None,
    documentation=None,
    covers=None,
    git_repo=None,
    cite=None,
    input_optional_parameters=True,
    model_postprocessing=None,
    for_deepimagej=False,
    links=None,
    maintainers=None,
    min_shape=None,
    halo=None,
    checkpoint_name="best",
    training_data=None,
    config={}
):
    """Export model to bioimage.io model format.
    """
    # Load the trainer and model.
    trainer = get_trainer(checkpoint, name=checkpoint_name, device="cpu")
    model, model_kwargs = _get_model(trainer, model_postprocessing)

    # Get input data from the trainer if it is not given.
    if input_data is None:
        input_data = _get_input_data(trainer)

    with tempfile.TemporaryDirectory() as export_folder:

        # Create the weight description.
        weight_description = _create_weight_description(model, export_folder, model_kwargs)

        # Create the test input/output files.
        test_in_paths, test_out_paths = _write_data(input_data, model, trainer, export_folder)
        # Get the descriptions for inputs, outputs and notebook links.
        input_description, output_description, notebook_link = _get_inout_descriptions(
            trainer, model, model_kwargs, test_in_paths, test_out_paths, min_shape, halo
        )

        # Get the additional kwargs.
        kwargs = _get_kwargs(
            trainer, name, description,
            authors, tags, license, documentation,
            git_repo, cite, maintainers,
            export_folder, input_optional_parameters
        )

        # TODO double check the current link policy
        # The apps to link with this model, by default ilastik.
        if links is None:
            links = []
        links.append("ilastik/ilastik")
        # add the notebook link, if available
        if notebook_link is not None:
            links.append(notebook_link)
        kwargs.update({"links": links})

        if covers is not None:
            kwargs["covers"] = covers

        model_description = spec.ModelDescr(
            inputs=input_description,
            outputs=output_description,
            weights=weight_description,
            config=config,
            **kwargs,
        )

        save_bioimageio_package(model_description, output_path=output_path)

    # Validate the model.
    val_success = _validate_model(output_path)
    if val_success:
        print(f"The model was successfully exported to '{output_path}'.")
    else:
        warn(f"Validation of the bioimageio model exported to '{output_path}' has failed. " +
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
    weight_spec = model_spec.weights.pytorch_state_dict
    model = PytorchModelAdapter.get_network(weight_spec)
    weight_file = weight_spec.source.path
    if not os.path.exists(weight_file):
        root_folder = f"{model_spec.root.filename}.unzip"
        assert os.path.exists(root_folder), root_folder
        weight_file = os.path.join(root_folder, weight_file)
    assert os.path.exists(weight_file), weight_file
    state = torch.load(weight_file, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def _load_normalizer(model_spec):
    inputs = model_spec.inputs[0]
    preprocessing = inputs.preprocessing

    # Filter out ensure dtype.
    preprocessing = [preproc for preproc in preprocessing if preproc.id != "ensure_dtype"]
    if len(preprocessing) == 0:
        return None

    ndim = len(inputs.axes) - 2
    shape = inputs.shape
    if hasattr(shape, "min"):
        shape = shape.min

    conf = preprocessing[0]
    name = conf.id
    spec_kwargs = conf.kwargs

    def _get_axis(axes):
        label_to_id = {"channel": 0, "z": 1, "y": 2, "x": 3} if ndim == 3 else\
            {"channel": 0, "y": 1, "x": 2}
        axis = tuple(label_to_id[ax] for ax in axes)

        # Is the axis full? Then we don't need to specify it.
        if len(axis) == ndim + 1:
            return None

        # Drop the channel axis if we have only a single channel.
        # Because torch_em squeezes the channel axis in this case.
        if shape[1] == 1:
            axis = tuple(ax - 1 for ax in axis if ax > 0)
        return axis

    axis = _get_axis(spec_kwargs.get("axes", None))
    if name == "zero_mean_unit_variance":
        kwargs = {"axis": axis}
        normalizer = functools.partial(torch_em.transform.raw.standardize, **kwargs)

    elif name == "fixed_zero_mean_unit_variance":
        kwargs = {"axis": axis, "mean": spec_kwargs["mean"], "std": spec_kwargs["std"]}
        normalizer = functools.partial(torch_em.transform.raw.standardize, **kwargs)

    elif name == "scale_linear":
        min_ = -spec_kwargs["offset"]
        max_ = 1. / spec_kwargs["gain"]
        kwargs = {"axis": axis, "minval": min_, "maxval": max_}
        normalizer = functools.partial(torch_em.transform.raw.normalize, **kwargs)

    elif name == "scale_range":
        assert spec_kwargs["mode"] == "per_sample"  # Can't parse the other modes right now.
        lower, upper = spec_kwargs["min_percentile"], spec_kwargs["max_percentile"]
        if np.isclose(lower, 0.0) and np.isclose(upper, 100.0):
            normalizer = functools.partial(torch_em.transform.raw.normalize, axis=axis)
        else:
            kwargs = {"axis": axis, "lower": lower, "upper": upper}
            normalizer = functools.partial(torch_em.transform.raw.normalize_percentile, **kwargs)

    else:
        msg = f"torch_em does not support the use of the biomageio preprocessing function {name}."
        raise RuntimeError(msg)

    return normalizer


def import_bioimageio_model(spec_path, return_spec=False, device="cpu"):
    model_spec = core.load_description(spec_path)

    model = _load_model(model_spec, device=device)
    normalizer = _load_normalizer(model_spec)

    if return_spec:
        return model, normalizer, model_spec
    else:
        return model, normalizer


# TODO
def import_trainer_from_bioimageio_model(spec_path):
    pass


# TODO: the weight conversion needs to be updated once the
# corresponding functionality in bioimageio.core is updated
#
# Weight Conversion
#


def _convert_impl(spec_path, weight_name, converter, weight_type, **kwargs):
    with tempfile.TemporaryDirectory() as tmp_dir:
        weight_path = os.path.join(tmp_dir, weight_name)
        model_spec = core.load_description(spec_path)
        weight_descr = converter(model_spec, weight_path, **kwargs)
        # TODO double check
        setattr(model_spec.weights, weight_type, weight_descr)
        save_bioimageio_package(model_spec, output_path=spec_path)


def convert_to_onnx(spec_path, opset_version=12):
    raise NotImplementedError
    # converter = weight_converter.convert_weights_to_onnx
    # _convert_impl(spec_path, "weights.onnx", converter, "onnx", opset_version=opset_version)
    # return None


def convert_to_torchscript(model_path):
    raise NotImplementedError
    # from bioimageio.core.weight_converter.torch._torchscript import convert_weights_to_torchscript

    # weight_name = "weights-torchscript.pt"
    # breakpoint()
    # _convert_impl(model_path, weight_name, convert_weights_to_torchscript, "torchscript")

    # # Check that we can load the converted weights.
    # model_spec = core.load_description(model_path)
    # weight_path = model_spec.weights.torchscript.weights
    # try:
    #     torch.jit.load(weight_path)
    #     return None
    # except Exception as e:
    #     return e


def add_weight_formats(model_path, additional_formats):
    for add_format in additional_formats:

        if add_format == "onnx":
            ret = convert_to_onnx(model_path)
        elif add_format == "torchscript":
            ret = convert_to_torchscript(model_path)

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
# Misc Functionality
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
