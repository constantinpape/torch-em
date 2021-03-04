import json
import os
from concurrent import futures

import luigi
import numpy as np
import nifty.tools as nt
import z5py

from cluster_tools.inference import InferenceLocal
from cluster_tools.inference.inference_embl import InferenceEmbl

OFFSETS = [
    [-1, 0, 0],
    [0, -1, 0],
    [0, 0, -1],
    [-2, 0, 0],
    [0, -3, 0],
    [0, 0, -3],
    [-3, 0, 0],
    [0, -9, 0],
    [0, 0, -9]
]


def update_block_shape(config_dir, block_shape, default_config):
    global_conf = os.path.join(config_dir, 'global.config')
    if os.path.exists(global_conf):
        with open(global_conf) as f:
            config = json.load(f)
    else:
        config = default_config

    if config['block_shape'] != block_shape:
        config['block_shape'] = block_shape

    with open(global_conf, 'w') as f:
        json.dump(config, f)


def predict(input_path, input_key,
            output_path, output_prefix,
            ckpt, gpus, tmp_folder, target,
            gpu_type='2080Ti', predict_affinities=False):
    task = InferenceLocal if target == 'local' else InferenceEmbl

    # halo = [8, 64, 64]
    # block_shape = [32, 256, 256]

    # larger halo
    halo = [12, 96, 96]
    block_shape = [24, 128, 128]

    if predict_affinities:
        output_key = {
            f'{output_prefix}/foreground': [0, 1],
            f'{output_prefix}/affinities': [1, 10]
        }
    else:
        output_key = {
            f'{output_prefix}/foreground': [0, 1],
            f'{output_prefix}/boundaries': [1, 2]
        }

    config_dir = os.path.join(tmp_folder, 'configs')
    os.makedirs(config_dir, exist_ok=True)
    update_block_shape(config_dir, block_shape, task.default_global_config())

    conf = task.default_global_config()
    conf.update({'block_shape': block_shape})
    with open(os.path.join(config_dir, 'global.config'), 'w') as f:
        json.dump(conf, f)

    if target == 'local':
        device_mapping = {ii: gpu for ii, gpu in enumerate(gpus)}
    else:
        device_mapping = None
    n_threads = 6

    conf = task.default_task_config()
    conf.update({
        'dtype': 'uint8',
        'device_mapping': device_mapping,
        'threads_per_job': n_threads,
        'mixed_precision': True,
        'gpu_type': gpu_type,
        'qos': 'high',
        'mem_limit': 24,
        'time_limit': 600
    })
    with open(os.path.join(config_dir, 'inference.config'), 'w') as f:
        json.dump(conf, f)

    t = task(tmp_folder=tmp_folder, config_dir=config_dir, max_jobs=len(gpus),
             input_path=input_path, input_key=input_key,
             output_path=output_path, output_key=output_key,
             checkpoint_path=ckpt, halo=halo,
             framework='pytorch')
    assert luigi.build([t], local_scheduler=True)
    update_block_shape(config_dir, [32, 256, 256], task.default_global_config())


def set_bounding_box(tmp_folder, bounding_box):
    config = InferenceLocal.default_global_config()
    config.update({
        'roi_begin': [bb.start for bb in bounding_box],
        'roi_end': [bb.stop for bb in bounding_box]
    })

    config_folder = os.path.join(tmp_folder, 'configs')
    os.makedirs(config_folder, exist_ok=True)
    config_file = os.path.join(config_folder, 'global.config')

    with open(config_file, 'w') as f:
        json.dump(config, f)


def get_checkpoint(checkpoint, use_best=False, is_affinity_model=False):
    if use_best:
        path = os.path.join(checkpoint, 'best.pt')
    else:
        path = os.path.join(checkpoint, 'latest.pt')

    n_out = 10 if is_affinity_model else 2
    if 'large' in checkpoint:
        model_kwargs = dict(
            scale_factors=[
                [1, 2, 2],
                [1, 2, 2],
                [2, 2, 2],
                [2, 2, 2],
                [2, 2, 2]
            ],
            in_channels=1,
            out_channels=n_out,
            initial_features=128,
            gain=2,
            pad_convs=True,
            final_activation='Sigmoid'
        )
    else:
        model_kwargs = dict(
            scale_factors=[
                [1, 2, 2],
                [1, 2, 2],
                [2, 2, 2],
                [2, 2, 2]
            ],
            in_channels=1,
            out_channels=n_out,
            initial_features=64,
            gain=2,
            pad_convs=True,
            final_activation='Sigmoid'
        )

    ckpt = {
        'class': ('mipnet.models.unet', 'AnisotropicUNet'),
        'kwargs': model_kwargs,
        'checkpoint_path': path,
        'model_state_key': 'model_state'
    }

    return ckpt


def run_multicut(path,
                 checkpoint_name,
                 target,
                 max_jobs,
                 tmp_folder,
                 beta):
    from cluster_tools.workflows import MulticutSegmentationWorkflow
    task = MulticutSegmentationWorkflow

    config_dir = os.path.join(tmp_folder, 'configs')
    configs = task.get_config()

    ws_config = configs['watershed']
    ws_config.update({
        "threshold": 0.25,
        'apply_dt_2d': True,
        'apply_filters_2d': True,
        'apply_ws_2d': False,
        'sigma_seeds': 2.6
    })
    with open(os.path.join(config_dir, 'watershed.config'), 'w') as f:
        json.dump(ws_config, f)

    cost_config = configs['probs_to_costs']
    cost_config.update({
        'beta': beta
    })
    with open(os.path.join(config_dir, 'probs_to_costs.config'), 'w') as f:
        json.dump(cost_config, f)

    bd_key = f'predictions/{checkpoint_name}/boundaries'
    node_labels_key = f'node_labels/{checkpoint_name}/multicut'
    ws_key = f'segmentation/{checkpoint_name}/watershed'
    seg_key = f'segmentation/{checkpoint_name}/multicut'

    t = task(target=target, max_jobs=max_jobs,
             tmp_folder=tmp_folder, config_dir=config_dir,
             input_path=path, input_key=bd_key,
             ws_path=path, ws_key=ws_key,
             problem_path=os.path.join(tmp_folder, 'data.n5'),
             node_labels_key=node_labels_key,
             output_path=path, output_key=seg_key)
    assert luigi.build([t], local_scheduler=True)


def run_mws(data_path, checkpoint_name,
            target, max_jobs, tmp_folder,
            threshold):
    fg_key = f'predictions/{checkpoint_name}/foreground'
    mask_key = f'predictions/{checkpoint_name}/mask'
    aff_key = f'predictions/{checkpoint_name}/affinities'
    seg_key = f'segmentation/{checkpoint_name}/mutex_watershed'

    from cluster_tools.thresholded_components.threshold import ThresholdLocal, ThresholdSlurm
    task = ThresholdLocal if target == 'local' else ThresholdSlurm
    config_dir = os.path.join(tmp_folder, 'configs')
    t = task(tmp_folder=tmp_folder, config_dir=config_dir, max_jobs=max_jobs,
             input_path=data_path, input_key=fg_key,
             output_path=data_path, output_key=mask_key,
             threshold=0.5)
    assert luigi.build([t], local_scheduler=True)

    from cluster_tools.mutex_watershed import MwsWorkflow
    task = MwsWorkflow

    config_dir = os.path.join(tmp_folder, 'configs')

    configs = task.get_config()
    conf = configs['mws_blocks']
    conf.update({
        'strides': [4, 4, 4],
        'randomize_strides': True
    })
    with open(os.path.join(config_dir, 'mws_blocks.config'), 'w') as f:
        json.dump(conf, f)

    conf = configs['block_edge_features']
    conf.update({
        'offsets': OFFSETS
    })
    with open(os.path.join(config_dir, 'block_edge_features.config'), 'w') as f:
        json.dump(conf, f)

    # TODO with halo?
    halo = None
    t = task(tmp_folder=tmp_folder, config_dir=config_dir,
             target=target, max_jobs=max_jobs,
             input_path=data_path, input_key=aff_key,
             output_path=data_path, output_key=seg_key,
             offsets=OFFSETS, halo=halo,
             mask_path=data_path, mask_key=mask_key,
             stitch_via_mc=True)
    assert luigi.build([t], local_scheduler=True)


def postprocess(path, checkpoint_name,
                seg_key, out_key,
                target, max_jobs, tmp_folder,
                size_threshold=250, threshold=None):
    from cluster_tools.postprocess import FilterByThresholdWorkflow
    from cluster_tools.postprocess import SizeFilterWorkflow

    fg_key = f'predictions/{checkpoint_name}/foreground'
    hmap_key = f'predictions/{checkpoint_name}/boundaries'

    config_dir = os.path.join(tmp_folder, 'configs')

    if threshold is not None:
        task = FilterByThresholdWorkflow
        t = task(target=target, max_jobs=max_jobs,
                 tmp_folder=tmp_folder, config_dir=config_dir,
                 input_path=path, input_key=fg_key,
                 seg_in_path=path, seg_in_key=seg_key,
                 seg_out_path=path, seg_out_key=out_key,
                 threshold=threshold)
        assert luigi.build([t], local_scheduler=True)
        seg_key = out_key

    if size_threshold is not None:
        task = SizeFilterWorkflow
        t = task(tmp_folder=tmp_folder, config_dir=config_dir,
                 target=target, max_jobs=max_jobs,
                 input_path=path, input_key=seg_key,
                 output_path=path, output_key=out_key,
                 hmap_path=path, hmap_key=hmap_key,
                 relabel=True, preserve_zeros=True,
                 size_threshold=size_threshold)
        assert luigi.build([t], local_scheduler=True)


# this deserves a cluster tools task
def affinity_to_boundary(data_path, prediction_prefix,
                         tmp_folder, target, max_jobs):
    aff_key = os.path.join(prediction_prefix, 'affinities')
    bd_key = os.path.join(prediction_prefix, 'boundaries')

    with z5py.File(data_path, 'a') as f:
        if bd_key in f:
            return

        ds_affs = f[aff_key]
        shape = ds_affs.shape[1:]
        chunks = ds_affs.chunks[1:]
        ds_bd = f.require_dataset(bd_key, shape=shape, chunks=chunks, compression='gzip',
                                  dtype=ds_affs.dtype)

        blocking = nt.blocking([0, 0, 0], shape, chunks)

        def _block(block_id):
            block = blocking.getBlock(block_id)
            bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))

            bb_affs = (slice(None),) + bb
            affs = ds_affs[bb_affs]

            bd = np.maximum(affs[1], affs[2])
            bd = np.maximum(bd, np.maximum(affs[4], affs[5]))
            ds_bd[bb] = bd.astype(ds_bd.dtype)

        with futures.ThreadPoolExecutor(8) as tp:
            tp.map(_block, range(blocking.numberOfBlocks))


def segment_with_boundaries(sample,
                            checkpoint,
                            target,
                            gpus,
                            max_jobs=32,
                            bounding_box=None,
                            beta=.5,
                            threshold=0.25,
                            only_prediction=False,
                            gpu_type='2080Ti',
                            is_affinity_model=False,
                            size_threshold=250):
    checkpoint_name = os.path.split(checkpoint)[1]

    data_path = os.path.join('./data', f'{sample}.n5')
    raw_key = 'raw'
    prediction_prefix = os.path.join('predictions', checkpoint_name)
    tmp_folder = os.path.join('./tmp_folders', f'tmp_{checkpoint_name}_{sample}')

    if bounding_box is not None:
        set_bounding_box(tmp_folder, bounding_box)

    ckpt = get_checkpoint(checkpoint,
                          is_affinity_model=is_affinity_model)
    predict(data_path, raw_key,
            data_path, prediction_prefix,
            ckpt, gpus, tmp_folder, target,
            gpu_type=gpu_type,
            predict_affinities=is_affinity_model)
    if only_prediction:
        return
    if is_affinity_model:
        affinity_to_boundary(data_path, prediction_prefix,
                             tmp_folder, target, max_jobs)

    run_multicut(data_path, checkpoint_name,
                 target, max_jobs, tmp_folder,
                 beta=beta)

    seg_key = f'segmentation/{checkpoint_name}/multicut'
    out_key = f'segmentation/{checkpoint_name}/multicut_postprocessed'

    postprocess(data_path, checkpoint_name,
                seg_key, out_key,
                target, max_jobs, tmp_folder,
                threshold=threshold,
                size_threshold=size_threshold)


def segment_with_affinities(sample,
                            checkpoint,
                            target,
                            gpus,
                            max_jobs=32,
                            bounding_box=None,
                            threshold=0.5,
                            only_prediction=False,
                            gpu_type='2080Ti',
                            size_threshold=250):
    checkpoint_name = os.path.split(checkpoint)[1]

    data_path = os.path.join('./data', f'{sample}.n5')
    raw_key = 'raw'
    prediction_prefix = os.path.join('predictions', checkpoint_name)
    tmp_folder = os.path.join('./tmp_folders', f'tmp_{checkpoint_name}_{sample}_mws')

    if bounding_box is not None:
        set_bounding_box(tmp_folder, bounding_box)

    ckpt = get_checkpoint(checkpoint,
                          is_affinity_model=True)
    predict(data_path, raw_key,
            data_path, prediction_prefix,
            ckpt, gpus, tmp_folder, target,
            gpu_type=gpu_type,
            predict_affinities=True)
    if only_prediction:
        return
    affinity_to_boundary(data_path, prediction_prefix,
                         tmp_folder, target, max_jobs)

    run_mws(data_path, checkpoint_name,
            target, max_jobs, tmp_folder,
            threshold=threshold)

    seg_key = f'segmentation/{checkpoint_name}/mutex_watershed'
    out_key = f'segmentation/{checkpoint_name}/mutex_watershed_postprocessed'

    postprocess(data_path, checkpoint_name,
                seg_key, out_key,
                target, max_jobs, tmp_folder,
                size_threshold=size_threshold)


if __name__ == '__main__':
    segment_with_affinities(
        'small',
        './checkpoints/affinity_model_default_human_rat',
        'local',
        gpus=[0, 1, 2, 3]
    )
