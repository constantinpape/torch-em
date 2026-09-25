import os
import numpy as np
import pandas as pd
import z5py
from segmentation_impl import segment_with_affinities, segment_with_boundaries


def compute_summaries(metric):
    def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
        p = metric.params
        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]

        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = metric.eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:, :, aind]
        else:
            # dimension of recall: [TxKxAxM]
            s = metric.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:, aind]
        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])

        return mean_s

    def _summarizeDets():
        stats = np.zeros(6)
        names = ['mean_ap', 'ap-.5', 'ap-.75', 'ap-.75_small', 'ap-.75_medium', 'ap-.75_large']
        stats[0] = _summarize(1)
        stats[1] = _summarize(1, iouThr=.5)
        stats[2] = _summarize(1, iouThr=.75)
        stats[3] = _summarize(1, areaRng='small', iouThr=.75)
        stats[4] = _summarize(1, areaRng='medium', iouThr=.75)
        stats[5] = _summarize(1, areaRng='large', iouThr=.75)
        return names, stats

    if not metric.eval:
        raise Exception('Please run accumulate() first')

    return _summarizeDets()


# the code looks pretty inefficient, I can probably compute the
# same with elf / cluster_tools stuff and do it out of core
def map3d_impl(seg, gt):
    from map3d.vol3d_eval import VOL3Deval
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
    v3dEval = VOL3Deval(result_p, result_fn, pred_score_sorted)

    v3dEval.params.areaRng = area_rng
    v3dEval.accumulate()
    v3dEval.summarize()

    names, stats = compute_summaries(v3dEval)
    return names, stats


def validate(checkpoint, sample, seg_name, seg_key):
    print("Validating", sample, seg_key, "...")
    path = f'./data/{sample}.n5'
    labels_key = 'labels'
    name = os.path.split(checkpoint)[1]

    bb = np.s_[:]
    # bb = np.s_[:25, :256, :256]
    with z5py.File(path, 'r') as f:
        ds_labels = f[labels_key]
        ds_labels.n_threads = 16
        ds_seg = f[seg_key]
        ds_seg.n_threads = 16

        labels = ds_labels[bb]
        seg = ds_seg[bb]

    names, stats = map3d_impl(labels, seg)
    data = np.array([name, sample, seg_name] + stats.tolist())

    result_table = './validation_results.csv'
    if os.path.exists(result_table):
        results = pd.read_csv(result_table)
        results = results.append(pd.DataFrame(data[None], columns=results.columns))
    else:
        columns = ['network', 'sample', 'method'] + names
        results = pd.DataFrame(data[None], columns=columns)

    results.to_csv(result_table, index=False)


def prep_affinity_cache(checkpoint, sample):
    checkpoint_name = os.path.split(checkpoint)[1]
    tmp_folder = os.path.join('./tmp_folders', f'tmp_{checkpoint_name}_{sample}_mws')
    os.makedirs(tmp_folder, exist_ok=True)
    inference_log = os.path.join(tmp_folder, 'inference.log')
    with open(inference_log, 'w'):
        pass


def segment_and_validate(checkpoint, samples, target, beta, gpus,
                         only_prediction=False, gpu_type='2080Ti'):
    checkpoint_name = os.path.split(checkpoint)[1]
    is_affinity_model = 'affinity' in checkpoint_name
    for sample in samples:
        segment_with_boundaries(sample, checkpoint, target=target,
                                beta=beta, gpus=gpus,
                                only_prediction=only_prediction,
                                gpu_type=gpu_type,
                                is_affinity_model=is_affinity_model)
        if only_prediction:
            continue
        if is_affinity_model:
            prep_affinity_cache(checkpoint, sample)
            segment_with_affinities(sample, checkpoint, target, gpus,
                                    gpu_type=gpu_type)

        validate(checkpoint, sample, seg_name='multicut',
                 seg_key=f'segmentation/{checkpoint_name}/multicut_postprocessed')
        if is_affinity_model:
            validate(checkpoint, sample, seg_name='mutex_watershed',
                     seg_key=f'segmentation/{checkpoint_name}/mutex_watershed_postprocessed')


def val_v1():
    checkpoints = ['./checkpoints/affinity_model_large_human_rat',
                   './checkpoints/affinity_model_large_train_on_val_human_rat']
    # checkpoints = ['./checkpoints/affinity_model_default_human_rat']
    samples = ['human_val', 'rat_val']
    beta = .5

    target = 'local'
    only_prediction = False

    gpus = list(range(2))
    gpu_type = 'A100'

    # gpus = [1, 2, 3, 5]
    # gpu_type = '2080Ti'

    for checkpoint in checkpoints:
        segment_and_validate(checkpoint, samples, target, beta,
                             gpus=gpus,
                             only_prediction=only_prediction, gpu_type=gpu_type)


if __name__ == '__main__':
    val_v1()
