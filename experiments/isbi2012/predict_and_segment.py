import argparse
import numpy as np
import torch
from elf.io import open_file

from train_boundaries_2d import get_model

# TODO
# - prediction in 3d
# - prediction with affinities
# - segmentation with multicut (for boundaries), mutex watershed (for affinities)


def predict_boundaries_2d(in_path, out_path, checkpoint, device=torch.device('cuda')):
    model = get_model()
    state = torch.load(checkpoint)['model_state']
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    with open_file(in_path, 'r') as f:
        raw = f['raw'][:]

    prediction = np.zeros_like(raw, dtype='float32')

    with torch.no_grad():
        for z in range(raw.shape[0]):
            input_ = raw[z].astype('float32') / 255.
            input_ = torch.from_numpy(input_[None, None]).to(device)
            pred = model(input_).cpu().numpy()[0, 0]
            prediction[z] = pred

    with open_file(out_path, 'a') as f:
        ds = f.require_dataset('boundaries', prediction.shape, compression='gzip', dtype='float32',
                               chunks=(1,) + prediction.shape[1:])
        ds[:] = prediction

    return prediction


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('-c', '--checkpoint', required=True)

    args = parser.parse_args()
    in_path = args.input
    out_path = args.output
    checkpoint = args.checkpoint
    predict_boundaries_2d(in_path, out_path, checkpoint)
