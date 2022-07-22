import sys
import torch
from elf.io import open_file
from torch_em.transform.raw import standardize

from train_embeddings import get_model


def load_raw():
    halo = [16, 128, 128]
    path = '/scratch/pape/mito_em/data/human_test.n5'
    with open_file(path, 'r') as f:
        raw = f['raw']
        bb = tuple(slice(sh // 2 - ha, sh // 2 + ha)
                   for sh, ha in zip(raw.shape, halo))
    return raw[bb], bb


def predict_embeddings():
    ckpt = torch.load('./checkpoints/embedding_model_default_human_rat/best.pt')
    state = ckpt['model_state']
    model = get_model(False)

    device = torch.device('cuda')
    model = model.to(device)
    model.load_state_dict(state)

    raw, bb = load_raw()
    input_ = standardize(raw)

    with torch.no_grad():
        input_ = torch.from_numpy(input_[None, None]).to(device)
        pred = model(input_)
        pred = pred.cpu().numpy().squeeze(0)
    print(pred.shape)

    with open_file('./data.h5', 'a') as f:
        f.create_dataset('raw', data=raw[bb], compression='gzip')
        f.create_dataset('embeddings', data=pred, compression='gzip')


def predict_affinities():
    sys.path.append('..')
    from train_affinities import get_model as get_aff_model

    ckpt = torch.load('../checkpoints/affinity_model_default_human_rat/best.pt')
    state = ckpt['model_state']
    model = get_aff_model(False)

    device = torch.device('cuda')
    model = model.to(device)
    model.load_state_dict(state)

    raw, _ = load_raw()
    input_ = standardize(raw)

    with torch.no_grad():
        input_ = torch.from_numpy(input_[None, None]).to(device)
        pred = model(input_)
        pred = pred.cpu().numpy().squeeze(0)
    print(pred.shape)

    with open_file('./data.h5', 'a') as f:
        f.create_dataset('affinities', data=pred, compression='gzip')


if __name__ == '__main__':
    # predict_embeddings()
    predict_affinities()
