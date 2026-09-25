import os
import z5py
from shutil import copytree, copyfile

ROOT = '/g/kreshuk/pape/Work/data/mito_em/data'
SCRATCH = '/scratch/pape/mito_em/data'


def create_file(out_path, ref_path):
    os.makedirs(out_path, exist_ok=True)
    copyfile(
        os.path.join(ref_path, 'attributes.json'),
        os.path.join(out_path, 'attributes.json')
    )


def copy_to_scratch(in_path, out_path, out_key):
    if out_key in z5py.File(out_path, 'r'):
        return

    in_key = 'setup0/timepoint0/s0'
    copytree(
        os.path.join(in_path, in_key),
        os.path.join(out_path, out_key)
    )


# copy training, test and val data to scratch
def prepare_scratch():
    os.makedirs(SCRATCH, exist_ok=True)

    for name in ('rat', 'human'):
        for split in ('train', 'val', 'test'):
            print("Copying", name, split)
            out_path = os.path.join(SCRATCH, f'{name}_{split}.n5')

            raw_path = os.path.join(ROOT, f'{name}_{split}', 'images', 'local', 'em-raw.n5')
            create_file(out_path, raw_path)
            copy_to_scratch(raw_path, out_path, 'raw')

            label_path = os.path.join(ROOT, f'{name}_{split}', 'images', 'local', 'em-mitos.n5')
            if os.path.exists(label_path):
                copy_to_scratch(label_path, out_path, 'labels')


def make_small_volume():
    in_path = './data/human_train.n5'
    f = z5py.File(in_path, 'r')
    ds_r = f['raw']
    ds_l = f['labels']

    halo = [32, 256, 256]
    shape = ds_r.shape
    bb = tuple(slice(sh // 2 - ha, sh // 2 + ha) for sh, ha in zip(shape, halo))

    raw = ds_r[bb]
    labels = ds_l[bb]

    out_path = './data/small.n5'
    with z5py.File(out_path, 'a') as f:
        f.create_dataset('raw', data=raw, compression='gzip', chunks=ds_r.chunks)
        f.create_dataset('labels', data=labels, compression='gzip', chunks=ds_l.chunks)


if __name__ == '__main__':
    prepare_scratch()
    # make_small_volume()
