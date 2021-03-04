import os
import numpy as np
import pandas as pd
import z5py
from mobie import add_segmentation
from mobie.metadata.image_dict import load_image_dict

ROOT = '/g/kreshuk/pape/Work/data/mito_em/data'
RESOLUTION = [.03, .008, .008]
SCALE_FACTORS = [[1, 2, 2], [1, 2, 2], [2, 2, 2], [2, 2, 2]]


def compute_object_scores(seg, gt):
    from map3d.vol3d_util import seg_iou3d_sorted

    ui, uc = np.unique(seg, return_counts=True)
    uc = uc[ui > 0]
    ui = ui[ui > 0]
    pred_score = np.ones((len(ui), 2), int)
    pred_score[:, 0] = ui
    pred_score[:, 1] = uc

    thres = [5e3, 1.5e4]
    area_rng = np.zeros((len(thres) + 2, 2), int)
    area_rng[0, 1] = 1e10
    area_rng[-1, 1] = 1e10
    area_rng[2:, 0] = thres
    area_rng[1:-1, 1] = thres

    result_p, result_fn, pred_score_sorted = seg_iou3d_sorted(seg, gt, pred_score, area_rng)

    seg_ids = result_p[:, 0].astype('uint32')
    ious = result_p[:, 4]

    best_scores = []
    worst_scores = []
    unique_seg_ids = np.unique(seg_ids)
    for seg_id in unique_seg_ids:
        this_scores = ious[seg_ids == seg_id]
        best_scores.append(np.max(this_scores))
        worst_scores.append(np.max(this_scores))

    return unique_seg_ids, np.array(best_scores), np.array(worst_scores)


def make_score_table(checkpoint, sample, seg_name):
    print("make score table ...")
    key = 'setup0/timepoint0/s0'

    checkpoint_name = os.path.split(checkpoint)[1]
    mobie_name = f'{checkpoint_name}_{seg_name}'
    path = os.path.join(ROOT, sample, 'images', 'local', f'{mobie_name}.n5')
    print("loading segmentation")
    with z5py.File(path, 'r') as f:
        ds = f[key]
        ds.n_threads = 8
        seg = ds[:]

    print("loading labels")
    gt_path = os.path.join(ROOT, sample, 'images', 'local', 'em-mitos.n5')
    with z5py.File(gt_path, 'r') as f:
        ds = f[key]
        ds.n_threads = 8
        gt = ds[:]

    unique_seg_ids, best_scores, worst_scores = compute_object_scores(seg, gt)
    data = np.concatenate([
        unique_seg_ids[:, None],
        best_scores[:, None],
        worst_scores[:, None]
    ], axis=1)
    columns = ['label_id', 'best_score', 'worst_score']
    tab = pd.DataFrame(data, columns=columns)

    table_path = os.path.join(ROOT, sample, 'tables', mobie_name, 'scores.csv')
    tab.to_csv(table_path, sep='\t', index=False)


def seg_to_mobie(checkpoint, sample, seg_name,
                 target='local', max_jobs=16):
    checkpoint_name = os.path.split(checkpoint)[1]

    input_path = f'data/{sample}.n5'
    input_key = f'segmentation/{checkpoint_name}/{seg_name}_postprocessed'
    with z5py.File(input_path, 'r') as f:
        chunks = f[input_key].chunks

    seg_name = f'{checkpoint_name}_{seg_name}'
    tmp_folder = f'./tmp_folders/mobie_{sample}_{seg_name}'

    im_dict = load_image_dict(
        os.path.join(ROOT, sample, 'images', 'images.json')
    )
    if seg_name in im_dict:
        return

    add_segmentation(input_path, input_key,
                     ROOT, sample, seg_name,
                     resolution=RESOLUTION, scale_factors=SCALE_FACTORS,
                     chunks=chunks, max_jobs=max_jobs, target=target,
                     tmp_folder=tmp_folder)


def checkpoint_to_mobie(checkpoint, seg_name,
                        with_scores=True, target='local', max_jobs=16):
    for sample in ('human_val', 'rat_val'):
        seg_to_mobie(checkpoint, sample, seg_name,
                     target=target, max_jobs=max_jobs)
        if with_scores:
            make_score_table(checkpoint, sample, seg_name)


if __name__ == '__main__':
    checkpoint_to_mobie(
        './checkpoints/affinity_model_large_human_rat',
        'mutex_watershed'
    )
