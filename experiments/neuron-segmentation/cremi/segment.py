#
# instance segmentation using the model from the bioimage.io model-zoo
# the volumes with training and test data are available at https://cremi.org/data/
#

import argparse
import os
import time

import h5py
import numpy as np


def _predict(in_path, out_path, model):
    # imports only necessary for prediction
    import xarray as xr
    from bioimageio.core import load_resource_description
    from bioimageio.core.prediction_pipeline import create_prediction_pipeline
    from bioimageio.core.prediction import predict_with_tiling

    with h5py.File(in_path, "r") as f:
        print("Loading raw data ...")
        raw = f["volumes/raw"][:]

    print("Start prediction ...")
    tiling = {
        "tile": {"x": 256, "y": 256, "z": 32},
        "halo": {"x": 16, "y": 16, "z": 4}
    }
    input_tensor = xr.DataArray(raw[None, None], dims=("b", "c", "z", "y", "x"))

    model_rdf = load_resource_description(model)
    offsets = model_rdf.config["mws"]["offsets"]

    pred_pp = create_prediction_pipeline(bioimageio_model=model_rdf)
    affs = predict_with_tiling(pred_pp, input_tensor, tiling)
    if isinstance(affs, list):
        assert len(affs) == 1
        affs = affs[0]
    affs = affs.squeeze()

    print("Save prediction ...")
    with h5py.File(out_path, "a") as f:
        ds = f.create_dataset("affinities", data=affs, compression="gzip")
        ds.attrs["offsets"] = offsets
    return affs, offsets


def _segment(affs, offsets, out_path, solver):
    # imports only necessary for segmentation
    import nifty
    from elf.segmentation.features import compute_grid_graph, compute_grid_graph_affinity_features
    from elf.segmentation.multicut import get_multicut_solver, compute_edge_costs, _to_objective
    from elf.segmentation.mutex_watershed import mutex_watershed_clustering

    print("Compute graph problem from affinities ...")
    t0 = time.time()
    shape = affs.shape[1:]
    grid_graph = compute_grid_graph(shape)
    local_uvs, local_weights = compute_grid_graph_affinity_features(grid_graph, affs[:3], offsets[:3])
    lr_uvs, lr_weights = compute_grid_graph_affinity_features(grid_graph, affs[3:], offsets[3:], strides=[1, 4, 4])
    # compute the multicut problem by concatenating local and lr edges
    edges = np.concatenate([local_uvs, lr_uvs], axis=0)
    graph = nifty.graph.undirectedGraph(grid_graph.numberOfNodes)
    graph.insertEdges(edges)
    costs = np.concatenate([local_weights, lr_weights], axis=0)
    costs = compute_edge_costs(costs)
    objective = _to_objective(graph, costs)
    print("... in", time.time() - t0, "s")

    print("Segment with", solver, "...")
    if solver == "mutex-watershed":
        t_solve = time.time()
        node_labels = mutex_watershed_clustering(local_uvs, lr_uvs, local_weights, lr_weights)
        t_solve = time.time() - t_solve
    elif solver.startswith("rama"):
        _, mode = solver.split("_")
        solver_impl = get_multicut_solver("rama")
        t_solve = time.time()
        node_labels = solver_impl(graph, costs, mode=mode)
        t_solve = time.time() - t_solve
    else:
        solver_impl = get_multicut_solver(solver)
        t_solve = time.time()
        node_labels = solver_impl(graph, costs)
        t_solve = time.time() - t_solve

    energy = objective.evalNodeLabels(node_labels)
    print("... in", t_solve, "s")
    print("... with energy:", energy)
    # TODO save segmentation to file


def segment(in_path, out_path, model, solver, crop):
    if crop:
        bb = np.s_[:, :50, :512, :512]
    else:
        bb = np.s_[:]

    affs = None
    if os.path.exists(out_path):
        with h5py.File(out_path, "r") as f:
            if "affinities" in f:
                print("Loading pre-computed prediction from", out_path, "...")
                ds = f["affinities"]
                affs = ds[bb]
                offsets = ds.attrs["offsets"]
    if affs is None:
        affs, offsets = _predict(in_path, out_path, model)
        affs = affs[bb]

    _segment(affs, offsets, out_path, solver)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)

    # TODO upload this model to bioimage.io and use model id
    default_model = "../../exported_models_mws/cremi/rdf.yaml"
    parser.add_argument("-m", "--model", default=default_model)
    parser.add_argument("-s", "--solver", default="mutex-watershed")
    parser.add_argument("-c", "--crop", default=1, type=int)
    args = parser.parse_args()

    segment(args.input, args.output, args.model, args.solver, bool(args.crop))


if __name__ == "__main__":
    main()
