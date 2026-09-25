import argparse
import os

from elf.io import open_file
from elf.segmentation.mutex_watershed import blockwise_mutex_watershed
# from elf.wrapper import ThresholdWrapper

from train_affinities import OFFSETS


# blockwise_mutex_watershed currently doesn't work out of core, so everything needs to be loaded into mem
# TODO implement an outof core version
def segment(input_path, input_prefix, output_path, output_key, n_workers):
    with open_file(input_path, 'r') as f, open_file(output_path, 'a') as f_out:

        ds_fg = f[os.path.join(input_prefix, 'foreground')]
        ds_fg.n_threads = n_workers

        ds_affs = f[os.path.join(input_prefix, 'affinities')]
        ds_affs.n_threads = n_workers
        print("Loading affinities ...")
        affs = ds_affs[:]

        print("Loading mask ...")
        mask = ds_fg[:] > 0.5
        strides = [4, 4, 4]

        print("Run mutex watershed ...")
        seg = blockwise_mutex_watershed(affs, OFFSETS, strides,
                                        block_shape=ds_fg.chunks,
                                        randomize_strides=True,
                                        mask=mask,
                                        n_threads=n_workers)

        print("Writing result ...")
        ds_out = f_out.require_dataset(output_key,
                                       shape=ds_fg.shape,
                                       chunks=ds_fg.chunks,
                                       compression='gzip',
                                       dtype='uint64')
        ds_out.n_threads = n_workers
        ds_out[:] = seg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--output', required=True)

    parser.add_argument('--input_prefix', default='prediction')
    parser.add_argument('--output_key', default='segmentation/mws')
    parser.add_argument('--n_workers', default=8, type=int)

    args = parser.parse_args()
    segment(args.input, args.input_prefix,
            args.output, args.output_key,
            args.n_workers)
