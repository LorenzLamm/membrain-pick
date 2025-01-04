from membrain_pick.dataloading.data_utils import (
    get_csv_data,
    store_array_in_star,
    load_mesh_from_hdf5,
    store_mesh_in_hdf5,
)
from membrain_pick.clustering.mean_shift_utils import MeanShiftForwarder
from membrain_pick.clustering.mean_shift_membrainv1 import MeanShift_clustering
import numpy as np
import torch
import os
import trimesh
from membrain_pick.orientation import orientation_from_mesh


def mean_shift_for_scores(
    positions: np.ndarray,
    scores: np.ndarray,
    bandwidth: float,
    max_iter: int,
    margin: float,
    device: str,
    score_threshold: float = 9.0,
    method: str = "membrain_pick",
):
    if method == "membrain_pick":
        ms_forwarder = MeanShiftForwarder(
            bandwidth=bandwidth, max_iter=max_iter, device=device, margin=margin
        )

        positions = torch.from_numpy(positions).to(device)
        scores = torch.from_numpy(scores).to(device)
        mask = scores < score_threshold
        if mask.sum() == 0:
            return np.zeros((0, 3)), np.array([])
        scores = scores[mask]
        positions = positions[mask]
        out = ms_forwarder.mean_shift_forward(positions, scores)
        out_pos = out[0]
        out_p_num = out[1]
    elif method == "membrainv1":
        ms_forwarder = MeanShift_clustering(pos_thres=score_threshold)
        out_pos, out_p_num = ms_forwarder.cluster_NN_output(
            positions, 10.0 - scores, bandwidth=bandwidth
        )
    else:
        raise ValueError("Unknown method for mean shift clustering.")
    print("Found", out_pos.shape[0], "clusters.")
    return out_pos, out_p_num


def mean_shift_for_h5(
    h5_file: str,
    out_dir: str,
    bandwidth: float,
    max_iter: int,
    margin: float,
    device: str,
    method: str = "membrain_pick",
    score_threshold: float = 9.0,
):
    mesh_data = load_mesh_from_hdf5(h5_file)
    verts = mesh_data["points"]
    scores = mesh_data["scores"]
    out_pos, _ = mean_shift_for_scores(
        verts, scores, bandwidth, max_iter, margin, device, score_threshold, method
    )
    # add positions to h5 container
    mesh_data["cluster_centers"] = out_pos
    out_file = os.path.join(out_dir, os.path.basename(h5_file))
    os.makedirs(out_dir, exist_ok=True)
    store_mesh_in_hdf5(
        out_file,
        **mesh_data,
    )

    store_clusters(h5_file, out_dir, out_pos, np.zeros((0,)), verts, mesh_data["faces"])


def mean_shift_for_csv(
    csv_file: str,
    out_dir: str,
    bandwidth: float,
    max_iter: int,
    margin: float,
    device: str,
    score_threshold: float = 9.0,
):
    csv_data = np.array(get_csv_data(csv_file), dtype=float)
    positions = csv_data[:, :3]
    scores = csv_data[:, 3]
    out_pos, out_p_num = mean_shift_for_scores(
        positions, scores, bandwidth, max_iter, margin, device, score_threshold
    )
    store_clusters(csv_file, out_dir, out_pos, out_p_num)


def store_clusters(
    csv_file: str,
    out_dir: str,
    out_pos: np.ndarray,
    out_p_num: np.ndarray,
    verts: np.ndarray = None,
    faces: np.ndarray = None,
):
    os.makedirs(out_dir, exist_ok=True)
    print("Clustering found", out_pos.shape[0], "clusters.")
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    if out_pos.shape[0] != 0:
        relion_euler_angles = orientation_from_mesh(out_pos, mesh)
    else:
        out_pos = np.zeros((0, 3))
        relion_euler_angles = np.zeros((0, 3))

    out_pos = np.concatenate([out_pos, relion_euler_angles], axis=1)
    # if csv_file ends with .csv, replace it with _clusters.star, also replace .h5 if it ends with that
    if csv_file.endswith(".csv"):
        out_file = os.path.join(
            out_dir, os.path.basename(csv_file).replace(".csv", "_clusters.star")
        )
    elif csv_file.endswith(".h5"):
        out_file = os.path.join(
            out_dir, os.path.basename(csv_file).replace(".h5", "_clusters.star")
        )
    store_array_in_star(
        out_file=out_file,
        data=out_pos,
        header=[
            "rlnCoordinateX",
            "rlnCoordinateY",
            "rlnCoordinateZ",
            "rlnAngleRot",
            "rlnAngleTilt",
            "rlnAnglePsi",
        ],
    )
