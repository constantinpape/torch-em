import os
from shutil import rmtree

import imageio
import numpy as np
import z5py

from mobie.import_data.util import downscale, add_max_id
from mobie.initialization import make_dataset_folders
from mobie.metadata import add_bookmark, add_dataset, add_to_image_dict
from mobie.tables import compute_default_table
from tqdm import tqdm

ROOT = '/g/kreshuk/pape/Work/data/mito_em'
MOBIE_ROOT = os.path.join(ROOT, 'data')
os.makedirs(MOBIE_ROOT, exist_ok=True)

RESOLUTION = [.03, .008, .008]
SCALE_FACTORS = [[1, 2, 2], [1, 2, 2], [2, 2, 2], [2, 2, 2]]


def get_slices(folder):
    files = os.listdir(folder)
    files.sort()
    files = [os.path.splitext(ff)[0] for ff in files]
    slice_ids = [int(ff[2:]) if ff.startswith('im') else int(ff[3:]) for ff in files]
    return slice_ids


def get_split(dataset, split):
    assert split in ('train', 'val', 'test')
    assert dataset in ('MitoEM-H', 'MitoEM-R')

    if split in ('train', 'val'):
        folder = os.path.join(ROOT, dataset, f'mito_{split}')
        slice_ids = get_slices(folder)
    else:
        non_test = get_split(dataset, 'train') + get_split(dataset, 'val')
        all_slice_ids = get_slices(os.path.join(ROOT, dataset, 'im'))
        slice_ids = list(set(all_slice_ids) - set(non_test))

    return slice_ids


def load_volume(slice_ids, folder, pattern):
    pattern = os.path.join(folder, pattern)

    im0 = pattern % slice_ids[0]
    im0 = imageio.imread(im0)

    shape = (len(slice_ids),) + im0.shape

    out = np.zeros(shape, dtype=im0.dtype)
    out[0] = im0

    for z, slice_id in tqdm(enumerate(slice_ids[1:], 1), total=len(slice_ids) - 1):
        out[z] = imageio.imread(pattern % slice_id)

    return out


def load_raw(dataset, split):
    slice_ids = get_split(dataset, split)
    folder = os.path.join(ROOT, dataset, 'im')
    pattern = 'im%04i.png'
    return load_volume(slice_ids, folder, pattern)


def load_seg(dataset, split):
    slice_ids = get_split(dataset, split)
    folder = os.path.join(ROOT, dataset, f'mito_{split}')
    pattern = 'seg%04i.tif'
    return load_volume(slice_ids, folder, pattern)


def check_volume(dataset, split):
    import napari
    raw = load_raw(dataset, split)
    seg = load_seg(dataset, split)

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(raw)
        viewer.add_labels(seg)


def make_raw(dataset, split, out_path):
    raw = load_raw(dataset, split)
    key = 'setup0/timepoint0/s0'
    chunks = (32, 256, 256)
    with z5py.File(out_path, 'a') as f:
        f.create_dataset(key, data=raw, compression='gzip',
                         chunks=chunks, n_threads=16)

    tmp_folder = './tmp_ds_raw'
    downscale(out_path, key, out_path,
              RESOLUTION, SCALE_FACTORS, chunks,
              tmp_folder, target='local', max_jobs=16,
              block_shape=chunks, library='skimage')


def make_seg(dataset, split, out_path):
    seg = load_seg(dataset, split)
    key = 'setup0/timepoint0/s0'
    chunks = (32, 256, 256)
    with z5py.File(out_path, 'a') as f:
        f.create_dataset(key, data=seg, compression='gzip',
                         chunks=chunks, n_threads=16)

    tmp_folder = './tmp_ds_seg'
    downscale(out_path, key, out_path,
              RESOLUTION, SCALE_FACTORS, chunks,
              tmp_folder, target='local', max_jobs=16,
              block_shape=chunks, library='vigra',
              library_kwargs={'order': 0})
    add_max_id(out_path, key, out_path, key,
               tmp_folder, target='local', max_jobs=16)


def make_dataset(dataset_prefix, dataset):
    for split in ('val', 'train', 'test'):
        dataset_name = f'{dataset_prefix}_{split}'
        dataset_folder = make_dataset_folders(ROOT, dataset_name)
        add_dataset(MOBIE_ROOT, dataset_name, is_default=True)

        raw_name = 'em-raw'
        raw_path = os.path.join(dataset_folder, 'images', 'local', raw_name + '.n5')
        make_raw(dataset, split, raw_path)
        xml_path = raw_path.replace('.n5', '.xml')
        add_to_image_dict(dataset_folder, 'image', xml_path, add_remote=False)

        if split in ('train', 'val'):
            seg_name = 'em-mitos'
            seg_path = os.path.join(dataset_folder, 'images', 'local', seg_name + '.n5')
            make_seg(dataset, split, seg_path)

            table_folder = os.path.join(dataset_folder, 'tables', seg_name)
            os.makedirs(table_folder, exist_ok=True)
            table_path = os.path.join(table_folder, 'default.csv')
            compute_default_table(seg_path, 'setup0/timepoint0/s0', table_path, RESOLUTION,
                                  tmp_folder='./tmp_ds_seg', target='local',
                                  max_jobs=16)

            xml_path = seg_path.replace('.n5', '.xml')
            add_to_image_dict(dataset_folder, 'segmentation', xml_path,
                              table_folder=table_folder)

        add_bookmark(dataset_folder, 'default', 'default',
                     layer_settings={raw_name: {'contrastLimits': [0., 255.]}})

        rmtree('./tmp_ds_raw')
        if os.path.exists('./tmp_ds_seg'):
            rmtree('./tmp_ds_seg')


if __name__ == '__main__':
    make_dataset('human', 'MitoEM-H')
    make_dataset('rat', 'MitoEM-R')
