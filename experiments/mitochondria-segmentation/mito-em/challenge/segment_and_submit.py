import os
import z5py
import h5py
from segmentation_impl import segment_with_affinities


def make_submission(checkpoint, seg_name):
    name = os.path.split(checkpoint)[1]
    out_folder = f'./submissions/{name}_{seg_name}'
    os.makedirs(out_folder, exist_ok=True)

    print("Make submission for", name)
    for ii, sample in enumerate(('human', 'rat')):
        in_path = f'./data/{sample}_test.n5'
        with z5py.File(in_path, 'r') as f:
            ds = f[f'segmentation/{name}/{seg_name}_postprocessed']
            ds.n_threads = 16
            seg = ds[:].astype('int64')

        out_path = os.path.join(out_folder, f'{ii}_{sample}_instance_seg_pred.h5')
        with h5py.File(out_path, 'a') as f:
            f.create_dataset('dataset_1', data=seg, compression='gzip')


def segment_and_submit(checkpoint, target, beta, only_prediction=False):
    samples = ['human_test', 'rat_test']
    is_affinity_model = 'affinity' in checkpoint
    assert is_affinity_model
    for sample in samples:
        segment_with_affinities(sample, checkpoint, target=target, gpus=list(range(4)),
                                gpu_type='A100', only_prediction=only_prediction)
    if only_prediction:
        return
    make_submission(checkpoint, 'mutex_watershed')


if __name__ == '__main__':
    checkpoints = [
        './checkpoints/affinity_model_large_human_rat',
        './checkpoints/affinity_model_large_train_on_val_human_rat'
    ]

    target = 'local'
    only_prediction = False

    beta = .5
    for checkpoint in checkpoints:
        segment_and_submit(checkpoint, target, beta, only_prediction=only_prediction)
