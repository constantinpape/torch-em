import os
import json
import subprocess
from shutil import copyfile

import imageio
import numpy as np
import torch
import torch_em
from elf.io import open_file

try:
    from bioimageio.spec import schema
    from bioimageio.spec.utils import yaml
    from bioimageio.spec.utils.build_spec import build_spec
except ImportError:
    build_spec = None


# try to load from filepath
def _get_trainer(trainer):
    # try to load from file
    if isinstance(trainer, str):
        assert os.path.exists(trainer)
        trainer = torch_em.trainer.DefaultTrainer.from_checkpoint(trainer)
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
            yaml.dump(dependencies, f)
    else:
        assert os.path.exists(dependencies)
        dep = yaml.load(dependencies)
        assert "channels" in dep
        assert "name" in dep
        assert "dependencies" in dep
        copyfile(dependencies, dep_path)


def _get_normalizer(trainer):
    dataset = trainer.train_loader.dataset
    if isinstance(dataset, torch_em.data.concat_dataset.ConcatDataset):
        dataset = dataset.datasets[0]
    # TODO the raw transform may contian multiple transformations beside the
    # normalization functions. Try to parse this to only return the normalization.
    preprocesser = dataset.raw_transform
    return preprocesser


def _write_data(input_data, model, trainer, export_folder):
    # normalize the input data if we have a normalization function
    normalizer = _get_normalizer(trainer)
    test_input = input_data if normalizer is None else normalizer(input_data)

    # pad to 4d/5d
    test_input = _pad(test_input, trainer)

    # run prediction
    with torch.no_grad():
        test_tensor = torch.from_numpy(test_input).to(trainer.device)
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
            call_res = subprocess.run(['git', 'remote', '-v'], capture_output=True)
            repo = call_res.stdout.decode('utf8').split('\n')[0].split()[1]
            repo = repo if repo else None
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
        'cite': _get_kwarg('cite', cite,
                           lambda: ['https://github.com/constantinpape/torch-em.git'],
                           is_list=True)
    }

    return kwargs


def _write_weights(model, export_folder):
    weights = model.state_dict()
    weight_path = os.path.join(export_folder, 'weights.pt')
    torch.save(weights, weight_path)
    return weight_path


def _create_cover(in_path, out_path):
    input_ = np.load(in_path)
    output = np.load(out_path)

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

    # TODO double check if this works
    def _grid_im(im0, im1):
        ims_per_row = 3
        n_chan = im1.shape[0]
        n_images = n_chan + 1
        n_rows = int(np.ceil(float(n_images) / ims_per_row))

        n, m = im_shape
        x, y = ims_per_row * n, n_rows * m
        out = np.zeros((3, x, y))
        images = [im0] + [np.repeat(im1[i:i+1], 3, axis=0) for i in range(n_chan)]

        i, j = 0, 0
        for im in images:
            x0, x1 = i * n, (i + 1) * n
            y0, y1 = j * m, (j + 1) * m
            out[:, x0:x1, y0:y1] = im

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


# TODO support conversion to onnx
# TODO more options for the bioimageio export:
# - preprocessing!
# - variable input / output shapes, halo
# - config for custom params (e.g. offsets for mws)
def export_biomageio_model(trainer, input_data, export_folder,
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
    if build_spec is None:
        raise RuntimeError("Need bioimageio package")

    # load trainer and model
    trainer = _get_trainer(trainer)
    model, model_kwargs = _get_model(trainer)

    # create the weights
    os.makedirs(export_folder, exist_ok=True)
    weight_path = _write_weights(model, export_folder)

    # create the test input/output file
    test_in_path, test_out_path = _write_data(input_data, model, trainer, export_folder)

    # create the model source file
    source = _write_source(model, export_folder)

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

    model_spec = build_spec(
        source=source,
        model_kwargs=model_kwargs,
        weight_uri=weight_path,
        weight_type="pytorch_state_dict",
        test_inputs=test_in_path,
        test_outputs=test_out_path,
        root=export_folder,
        dependencies="conda:./environment.yaml",
        covers=cover_path,
        **kwargs
    )

    serialized = schema.Model().dump(model_spec)
    name = trainer.name if name is None else name
    out_path = os.path.join(export_folder, f'{name}.model.yaml')
    with open(out_path, 'w') as f:
        yaml.dump(serialized, f)

    # TODO load and validate the model
    # TODO print links for how to use the export


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
