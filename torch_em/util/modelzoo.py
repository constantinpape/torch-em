import functools
import json
import os
import pathlib
import subprocess
from shutil import copyfile
from warnings import warn

import imageio
import numpy as np
import torch
import torch_em
from elf.io import open_file

try:
    from bioimageio import spec
except ImportError:
    spec = None


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


# try to load from filepath
def _get_trainer(checkpoint, name='best', device=None):
    # try to load from file
    if isinstance(checkpoint, str):
        assert os.path.exists(checkpoint)
        trainer = torch_em.trainer.DefaultTrainer.from_checkpoint(checkpoint,
                                                                  name=name,
                                                                  device=device)
    else:
        trainer = checkpoint
    assert isinstance(trainer, torch_em.trainer.DefaultTrainer)
    return trainer


def _get_model(trainer):
    model = trainer.model
    model.eval()
    model_kwargs = model.init_kwargs
    # clear the kwargs of non builtins
    # TODO warn if we strip any non-standard arguments
    model_kwargs = {k: v for k, v in model_kwargs.items()
                    if not isinstance(v, type)}
    return model, model_kwargs


def _pad(input_data, trainer):
    ndim = trainer.train_loader.dataset.ndim
    target_dims = ndim + 2
    for _ in range(target_dims - input_data.ndim):
        input_data = np.expand_dims(input_data, axis=0)
    return input_data


def _write_depedencies(export_folder, dependencies):
    dep_path = os.path.join(export_folder, 'environment.yaml')
    if dependencies is None:
        dependencies = {
            "channels": ["pytorch", "conda-forge"],
            "name": "torch-em-deploy",
            "dependencies": "pytorch"
        }
        with open(dep_path, 'w') as f:
            spec.utils.yaml.dump(dependencies, f)
    else:
        assert os.path.exists(dependencies)
        dep = spec.utils.yaml.load(dependencies)
        assert "channels" in dep
        assert "name" in dep
        assert "dependencies" in dep
        copyfile(dependencies, dep_path)


def _get_normalizer(trainer):
    dataset = trainer.train_loader.dataset
    if isinstance(dataset, torch_em.data.concat_dataset.ConcatDataset):
        dataset = dataset.datasets[0]
    preprocesser = dataset.raw_transform
    try:
        normalizer = preprocesser.normalizer
        return normalizer
    except AttributeError:
        warn("Could not parse the normalization function, 'preprocessing' field will be empty.")
        return preprocesser


def _write_data(input_data, model, trainer, export_folder):
    # normalize the input data if we have a normalization function
    normalizer = _get_normalizer(trainer)

    # pad to 4d/5d and normalize the input data
    # NOTE we have to save the padded data, but without normalization
    test_input = _pad(input_data, trainer)
    normalized = normalize_with_batch(test_input, normalizer)

    # run prediction
    with torch.no_grad():
        test_tensor = torch.from_numpy(normalized).to(trainer.device)
        test_output = model(test_tensor).cpu().numpy()

    # save the input / output
    test_in_path = os.path.join(export_folder, 'test_input.npy')
    np.save(test_in_path, test_input)
    test_out_path = os.path.join(export_folder, 'test_output.npy')
    np.save(test_out_path, test_output)
    return test_in_path, test_out_path


def _write_source(model, export_folder):
    # copy the model source file if it's a torch_em model
    # (for now only u-net). otherwise just put the full python class
    module = str(model.__class__.__module__)
    cls_name = str(model.__class__.__name__)
    if module == 'torch_em.model.unet':
        source_path = os.path.join(
            os.path.split(__file__)[0],
            '../model/unet.py'
        )
        source_target_path = os.path.join(export_folder, 'unet.py')
        copyfile(source_path, source_target_path)
        source = f'./unet.py::{cls_name}'
    else:
        source = f"{source}.{cls_name}"
    return source


def _get_kwargs(trainer, name, description,
                authors, tags,
                license, documentation,
                git_repo, cite,
                export_folder, input_optional_parameters):
    if input_optional_parameters:
        print("Enter values for the optional parameters.")
        print("If the default value in [] is satisfactory, press enter without additional input.")
        print("Please enter lists using json syntax.")

    def _get_kwarg(kwarg_name, val, default, is_list=False, fname=None):
        # if we don't have a value, we either ask user for input (offering the default)
        # or just use the default if input_optional_parameters is False
        if val is None and input_optional_parameters:
            default_val = default()
            choice = input(f"{kwarg_name} [{default_val}]: ")
            val = choice if choice else default_val
        elif val is None:
            val = default()

        if fname is not None:
            save_path = os.path.join(export_folder, fname)
            with open(save_path, 'w') as f:
                f.write(val)
            return f'./{fname}'

        if is_list and isinstance(val, str):
            val = val.replace("'", '"')  # enable single quotes
            val = json.loads(val)
        if is_list:
            assert isinstance(val, (list, tuple))
        return val

    def _default_authors():
        # first try to derive the author name from git
        try:
            call_res = subprocess.run(['git', 'config', 'user.name'], capture_output=True)
            author = call_res.stdout.decode('utf8').rstrip('\n')
            author = author if author else None  # in case there was no error, but output is empty
        except Exception:
            author = None

        # otherwise use the username
        if author is None:
            author = os.uname()[1]

        return [author]

    def _default_repo():
        try:
            call_res = subprocess.run(["git", "remote", "-v"], capture_output=True)
            repo = call_res.stdout.decode("utf8").split("\n")[0].split()[1]
            if repo:
                repo = repo.replace("git@github.com:", "https://github.com/")
            else:
                repo = None
        except Exception:
            repo = None
        return repo

    # TODO derive better default values:
    # - description: derive something from trainer.ndim, trainer.loss, trainer.model, ...
    # - tags: derive something from trainer.ndim, trainer.loss, trainer.model, ...
    # - documentation: derive something from trainer.ndim, trainer.loss, trainer.model, ...
    # - cite: make doi for torch_em and add it instead of url + derive citation from model
    kwargs = {
        'name': _get_kwarg('name', name, lambda: trainer.name),
        'description': _get_kwarg('description', name, lambda: trainer.name),
        'authors': _get_kwarg('authors', authors, _default_authors, is_list=True),
        'tags': _get_kwarg('tags', tags, lambda: [trainer.name], is_list=True),
        'license': _get_kwarg('license', license, lambda: 'MIT'),
        'documentation': _get_kwarg('documentation', documentation, lambda: trainer.name,
                                    fname='documentation.md'),
        'git_repo': _get_kwarg('git_repo', git_repo, _default_repo),
        'cite': _get_kwarg('cite', cite, lambda: {'training library': 'https://github.com/constantinpape/torch-em.git'})
    }

    return kwargs


def _write_weights(model, export_folder):
    weights = model.state_dict()
    weight_name = 'weights.pt'
    weight_path = os.path.join(export_folder, weight_name)
    torch.save(weights, weight_path)
    return f'./{weight_name}'


def _create_cover(in_path, out_path):
    input_ = np.load(in_path)
    axis = (0, 2, 3) if input_.ndim == 4 else (0, 2, 3, 4)
    input_ = torch_em.transform.raw.normalize(input_, axis=axis)

    output = np.load(out_path)
    axis = (0, 2, 3) if output.ndim == 4 else (0, 2, 3, 4)
    output = torch_em.transform.raw.normalize(output, axis=axis)

    def _to_image(data):
        assert data.ndim in (4, 5)
        if data.ndim == 5:
            z0 = data.shape[2] // 2
            data = data[0, :, z0]
        else:
            data = data[0, :]
        data = (data * 255).astype('uint8')
        return data

    input_ = _to_image(input_)
    output = _to_image(output)

    chan_in = input_.shape[0]
    # make sure the input is rgb
    if chan_in == 1:  # single channel -> repeat it 3 times
        input_ = np.repeat(input_, 3, axis=0)
    elif chan_in != 3:  # != 3 channels -> take first channe and repeat it 3 times
        input_ = np.repeat(input_[0:1], 3, axis=0)

    im_shape = input_.shape[1:]
    if im_shape != output.shape[1:]:  # just return the input image if shapes don't agree
        return input_

    def _diagonal_split(im0, im1):
        assert im0.shape[0] == im1.shape[0] == 3
        n, m = im_shape
        out = np.ones((3, n, m), dtype='uint8')
        im0_indices = np.tril_indices(n=n, m=m)
        im1_indices = np.triu_indices(n=n, m=m)
        out[:, im0_indices] = im0[:, im0_indices]
        out[:, im1_indices] = im1[:, im1_indices]
        return out

    def _grid_im(im0, im1):
        ims_per_row = 3
        n_chan = im1.shape[0]
        n_images = n_chan + 1
        n_rows = int(np.ceil(float(n_images) / ims_per_row))

        n, m = im_shape
        x, y = ims_per_row * n, n_rows * m
        out = np.zeros((3, y, x))
        images = [im0] + [np.repeat(im1[i:i+1], 3, axis=0) for i in range(n_chan)]

        i, j = 0, 0
        for im in images:
            x0, x1 = i * n, (i + 1) * n
            y0, y1 = j * m, (j + 1) * m
            out[:, y0:y1, x0:x1] = im

            i += 1
            if i == ims_per_row:
                i = 0
                j += 1

        return out

    chan_out = output.shape[0]
    if chan_out == 1:  # single prediction channel: create diagonal split
        im = _diagonal_split(input_, np.repeat(output, 3, axis=0))
    elif chan_out == 3:  # three prediction channel: create diagonal split with rgb
        im = _diagonal_split(input_, output)
    else:  # otherwise create grid image
        im = _grid_im(input_, output)

    # to channel last
    return im.transpose((1, 2, 0))


def _write_covers(test_in_path, test_out_path, export_folder, covers):
    if covers is None:  # generate a cover from the test input/output
        cover_path = ['./cover.jpg']
        cover_out = os.path.join(export_folder, 'cover.jpg')
        cover_im = _create_cover(test_in_path, test_out_path)
        imageio.imwrite(cover_out, cover_im)
    else:  # cover images were passed, copy them to the export folder
        cover_path = []
        for path in covers:
            assert os.path.exists(path)
            fname = os.path.split(path)[1]
            copyfile(path, os.path.join(export_folder, fname))
            cover_path.append(f'./{fname}')
    return cover_path


def _get_preprocessing(trainer):
    ndim = trainer.train_loader.dataset.ndim
    normalizer = _get_normalizer(trainer)

    if isinstance(normalizer, functools.partial):
        kwargs = normalizer.keywords
        normalizer = normalizer.func
    else:
        kwargs = {}

    def _get_axes(axis):
        if axis is None:
            axes = "cyx" if ndim == 2 else "czyx"
        else:
            axes_labels = ['c', 'y', 'x'] if ndim == 2 else ['c', 'z', 'y', 'x']
            axes = ''.join(axes_labels[i] for i in axis)
        return axes

    name = f"{normalizer.__module__}.{normalizer.__name__}"
    if name == 'torch_em.transform.raw.normalize':

        min_, max_ = kwargs.get('minval', None), kwargs.get('maxval', None)
        axes = _get_axes(kwargs.get('axis', None))
        assert (min_ is None) == (max_ is None)

        if min_ is None:
            preprocessing = {
                "name": "scale_range",
                "kwargs": {
                    "mode": 'per_sample',
                    "axes": axes,
                    "min_percentile": 0.0,
                    "max_percentile": 100.0
                }
            }
        else:
            preprocessing = {
                "name": "scale_linear",
                "kwargs": {
                    "gain": 1. / max_,
                    "offset": -min_,
                    "axes": axes
                }
            }

    elif name == 'torch_em.transform.raw.standardize':

        mean, std = kwargs.get('mean', None), kwargs.get('std', None)
        mode = 'per_sample' if mean is None else 'fixed'
        axes = _get_axes(kwargs.get('axis', None))
        preprocessing = {
            "name": "zero_mean_unit_variance",
            "kwargs": {
                "mode": mode,
                "axes": axes
            }
        }
        if mean is not None:
            preprocessing['kwargs']['mean'] = mean
        if std is not None:
            preprocessing['kwargs']['std'] = std

    elif name == 'torch_em.transform.normalize_percentile':

        lower, upper = kwargs.get('lower', 1.0), kwargs.get('upper', 99.0)
        axes = _get_axes(kwargs.get('axis', None))
        preprocessing = {
            "name": "scale_range",
            "kwargs": {
                "mode": 'per_sample',
                "axes": axes,
                "min_percentile": lower,
                "max_percentile": upper
            }
        }

    else:
        return None

    return [preprocessing]


def _get_tensor_kwargs(model, model_kwargs):
    module = str(model.__class__.__module__)
    name = str(model.__class__.__name__)
    # can derive tensor kwargs only for known torch_em models (only unet for now)
    if module == 'torch_em.model.unet':
        inc, outc = model_kwargs['in_channels'], model_kwargs['out_channels']
        if name == "UNet2d":
            depth = model_kwargs['depth']
            step = [0, 0] + [2 ** depth] * 2
            min_shape = [1, inc] + [2 ** (depth + 1)] * 2
            halo = [0, 0] + [2 ** (depth - 1)] * 2
        elif name == "UNet3d":
            depth = model_kwargs['depth']
            step = [0, 0] + [2 ** depth] * 3
            min_shape = [1, inc] + [2 ** (depth + 1)] * 3
            halo = [0, 0] + [2 ** (depth - 1)] * 3
        elif name == "AnisotropicUNet":
            scale_factors = model_kwargs['scale_factors']
            scale_prod = [
                int(np.prod(scale_factors[i][d] for i in range(len(scale_factors))))
                for d in range(3)
            ]
            assert len(scale_prod) == 3
            step = [0, 0] + scale_prod
            min_shape = [1, inc] + [2 * sp for sp in scale_prod]
            halo = [0, 0] + [sp // 2 for sp in scale_prod]
        else:
            raise RuntimeError(f"Cannot derive tensor parameters for {module}.{name}")

        ref = "input"
        if inc == outc:
            scale = [1] * len(step)
            offset = [0] * len(step)
        else:
            scale = [1, 0] + ([1] * (len(step) - 2))
            offset = [0, outc] + ([0] * (len(step) - 2))
        tensor_kwargs = {
            "input_step": step,
            "input_min_shape": min_shape,
            "output_reference": ref,
            "output_scale": scale,
            "output_offset": offset,
            "halo": halo
        }
        return tensor_kwargs
    else:
        return {}


def _validate_model(spec_path):
    model, normalizer, bio_spec = import_bioimageio_model(spec_path, return_spec=True)

    for test_in, test_out in zip(bio_spec.test_inputs, bio_spec.test_outputs):
        input_, expected = np.load(test_in), np.load(test_out)
        input_ = normalize_with_batch(input_, normalizer)
        with torch.no_grad():
            input_ = torch.from_numpy(input_)
            output = model(input_).numpy()
        if not np.allclose(expected, output):
            return False

    return True


#
# model export functionality
#


# TODO support conversion to onnx
# TODO config: training details derived from loss and optimizer, custom params, e.g. offsets for mws
def export_biomageio_model(checkpoint, input_data, export_folder,
                           dependencies=None, name=None,
                           description=None, authors=None,
                           tags=None, license=None,
                           documentation=None, covers=None,
                           git_repo=None, cite=None,
                           input_optional_parameters=True):
    """
    """

    # TODO update the error message to point
    # to the source for the bioimageio package
    if spec is None:
        raise RuntimeError("Need bioimageio package")

    # load trainer and model
    trainer = _get_trainer(checkpoint, device='cpu')
    model, model_kwargs = _get_model(trainer)

    # create the weights
    os.makedirs(export_folder, exist_ok=True)
    weight_path = _write_weights(model, export_folder)

    # create the test input/output file
    test_in_path, test_out_path = _write_data(input_data, model, trainer, export_folder)

    # create the model source file
    source = _write_source(model, export_folder)

    # derive the tensor kwargs from the model and its kwargs
    tensor_kwargs = _get_tensor_kwargs(model, model_kwargs)

    # create dependency file
    _write_depedencies(export_folder, dependencies)

    # create cover image
    cover_path = _write_covers(test_in_path, test_out_path, export_folder, covers)

    # get the additional kwargs
    kwargs = _get_kwargs(trainer, name, description,
                         authors, tags,
                         license, documentation,
                         git_repo, cite,
                         export_folder, input_optional_parameters)
    kwargs.update(tensor_kwargs)
    preprocessing = _get_preprocessing(trainer)

    model_spec = spec.utils.build_spec(
        source=source,
        model_kwargs=model_kwargs,
        weight_uri=weight_path,
        weight_type="pytorch_state_dict",
        test_inputs=[f'./{os.path.split(test_in_path)[1]}'],
        test_outputs=[f'./{os.path.split(test_out_path)[1]}'],
        root=export_folder,
        dependencies="conda:./environment.yaml",
        covers=cover_path,
        preprocessing=preprocessing,
        **kwargs
    )

    serialized = spec.schema.Model().dump(model_spec)
    name = trainer.name if name is None else name
    out_path = os.path.join(export_folder, f'{name}.model.yaml')
    with open(out_path, 'w') as f:
        spec.utils.yaml.dump(serialized, f)

    # load and validate the model
    val_success = _validate_model(out_path)

    if val_success:
        # TODO print links for how to use the export
        print(f"The model was successfully exported to '{export_folder}'.")
    else:
        warn(f"Validation of the bioimageio model exported to '{export_folder}' has failed. " +
             "You can use this model, but it will probably yield incorrect results.")
    return val_success


# TODO support bounding boxes
def _load_data(path, key):
    if key is None:
        ext = os.path.splitext(path)[-1]
        if ext == '.npy':
            return np.load(path)
        else:
            return imageio.imread(path)
    else:
        return open_file(path, mode='r')[key][:]


def main():
    import argparse
    parser = argparse.ArgumentParser(
        "Export model trained with torch_em to biomage.io model format"
    )
    parser.add_argument('-p', '--path', required=True,
                        help="Path to the checkpoint")
    parser.add_argument('-d', '--data', required=True,
                        help="")
    parser.add_argument('-f', '--export_folder', required=True,
                        help="")
    parser.add_argument('-k', '--key', default=None,
                        help="")

    args = parser.parse_args()
    export_biomageio_model(
        args.path, _load_data(args.data, args.key), args.export_folder
    )


#
# model import functionality
#

def _load_model(bio_spec):
    model = spec.utils.get_instance(bio_spec)
    weights = bio_spec.weights["pytorch_state_dict"]
    state = torch.load(weights.source, map_location='cpu')
    model.load_state_dict(state)
    model.eval()
    return model


def _load_normalizer(bio_spec):
    inputs = bio_spec.inputs[0]
    preprocessing = inputs.preprocessing
    if len(preprocessing) == 0:
        return None

    ndim = len(inputs.axes) - 2
    shape = inputs.shape
    if hasattr(shape, 'min'):
        shape = shape.min

    conf = preprocessing[0]
    name = conf.name
    spec_kwargs = conf.kwargs

    def _get_axis(axes):
        label_to_id = {'c': 0, 'z': 1, 'y': 2, 'x': 3} if ndim == 3 else\
            {'c': 0, 'y': 1, 'x': 2}
        axis = tuple(label_to_id[ax] for ax in axes)

        # is the axis full? Then we don't need to specify it.
        if len(axis) == ndim + 1:
            return None

        # drop the channel axis if we have only a single channel
        # (because torch_em squeezes the channel axis in this case)
        if shape[1] == 1:
            axis = tuple(ax - 1 for ax in axis if ax > 0)
        return axis

    if name == 'zero_mean_unit_variance':
        mode = spec_kwargs['mode']
        if mode == 'fixed':
            kwargs = {'mean': spec_kwargs['mean'], 'std': spec_kwargs['std']}
        else:
            axis = _get_axis(spec_kwargs['axes'])
            kwargs = {'axis': axis}
        normalizer = functools.partial(
            torch_em.transform.raw.standardize,
            **kwargs
        )

    elif name == 'scale_linear':
        axis = _get_axis(spec_kwargs['axes'])
        min_ = -spec_kwargs['offset']
        max_ = 1. / spec_kwargs['gain']
        kwargs = {'axis': axis, 'minval': min_, 'maxval': max_}
        normalizer = functools.partial(
            torch_em.transform.raw.normalize,
            **kwargs
        )

    elif name == 'scale_range':
        assert spec_kwargs.mode == 'per_sample'  # can't parse the other modes right now
        axis = _get_axis(spec_kwargs['axes'])
        lower, upper = spec_kwargs['min_percentile'], spec_kwargs['max_percentile']
        if np.isclose(lower, 0.0) and np.isclose(upper, 100.0):
            normalizer = functools.partial(
                torch_em.transform.raw.normalize,
                axis=axis
            )
        else:
            kwargs = {'axis': axis, 'lower': lower, 'upper': upper}
            normalizer = functools.partial(
                torch_em.transform.raw.normalize_percentile,
                **kwargs
            )

    else:
        msg = f"torch_em does not support the use of the biomageio preprocessing function {name}"
        raise RuntimeError(msg)

    return normalizer


def import_bioimageio_model(spec_path, return_spec=False):
    # TODO update the error message to point
    # to the source for the bioimageio package
    if spec is None:
        raise RuntimeError("Need bioimageio package")
    bio_spec = spec.load_and_resolve_spec(pathlib.Path(spec_path).absolute())

    model = _load_model(bio_spec)
    normalizer = _load_normalizer(bio_spec)

    if return_spec:
        return model, normalizer, bio_spec
    else:
        return model, normalizer


# TODO
def import_trainer_from_bioimageio_model(spec_path):
    pass
