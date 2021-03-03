import argparse
import os
import numpy as np
import torch

from elf.io import open_file
from torch_em.util.prediction import predict_with_halo
from train_affinities import get_model


def predict_affinities(checkpoint, gpu_ids,
                       input_path, input_key,
                       output_path, output_key):
    model = get_model()
    state_dict = torch.load(checkpoint)['model_state']
    model.load_state_dict(state_dict)

    block_shape = (96, 96, 96)
    halo = (32, 32, 32)

    with open_file(input_path, 'r') as f_in, open_file(output_path, 'a') as f_out:
        ds_in = f_in[input_key]
        shape = ds_in.shape

        ds_fg = f_out.require_dataset(os.path.join(output_key, 'foreground'),
                                      shape=shape, chunks=block_shape,
                                      compression='gzip', dtype='float32')

        aff_shape = (model.out_channels - 1,) + shape
        ds_affs = f_out.require_dataset(os.path.join(output_key, 'affinities'),
                                        shape=aff_shape, chunks=(1,) + block_shape,
                                        compression='gzip', dtype='float32')

        outputs = [(ds_fg, np.s_[0]), (ds_affs, np.s_[1:])]

        predict_with_halo(ds_in, model, gpu_ids, block_shape, halo,
                          output=outputs)


# example call for this script:
# python predict_affinities.py -c checkpoints/affinity-model/best.pt -i /g/kreshuk/pape/Work/data/platy_training_data/mitos/10nm/gt000/raw.h5 -o prediction.n5
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', '-c', required=True)
    parser.add_argument('--input', '-i', required=True)
    parser.add_argument('--output', '-o', required=True)

    parser.add_argument('--input_key', default='data')
    parser.add_argument('--output_key', default='prediction')
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0])

    args = parser.parse_args()
    predict_affinities(args.checkpoint, args.gpu_ids,
                       args.input, args.input_key,
                       args.output, args.output_key)
