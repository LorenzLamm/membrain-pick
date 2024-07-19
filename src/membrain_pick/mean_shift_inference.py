from membrain_pick.dataloading.data_utils import get_csv_data, store_array_in_csv, store_array_in_npy, store_point_and_vectors_in_vtp, store_array_in_star
from membrain_pick.optimization.mean_shift_utils import MeanShiftForwarder
from membrain_pick.mean_shift_membrainv1 import MeanShift_clustering
import numpy as np
import torch
import os
from scipy.spatial import distance_matrix


def mean_shift_for_scores(
        positions: np.ndarray,
        scores: np.ndarray,
        bandwidth: float,
        max_iter: int,
        margin: float,
        device: str,
        score_threshold: float = 9.0,
):
    

    # ms_forwarder = MeanShift_clustering(pos_thres=score_threshold)
    # clusters, cluster_labels = ms_forwarder.cluster_NN_output(positions, scores, bandwidth=bandwidth)
    # return clusters, cluster_labels

    ms_forwarder = MeanShiftForwarder(bandwidth=bandwidth,
                                  max_iter=max_iter,
                                  device=device,
                                  margin=margin)
    
    
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

    return out_pos, out_p_num

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
    out_pos, out_p_num = mean_shift_for_scores(positions, scores, bandwidth, max_iter, margin, device, score_threshold)
    store_clusters(csv_file, out_dir, out_pos, out_p_num)


def store_clusters(
        csv_file: str,
        out_dir: str,
        out_pos: np.ndarray,
        out_p_num: np.ndarray,
):
    os.makedirs(out_dir, exist_ok=True)
    # store_array_in_csv(
    #     out_file=os.path.join(out_dir, os.path.basename(csv_file).replace('.csv', '_clusters.csv')),
    #     data=out_pos, header=["x", "y", "z"]
    # )
    # store_array_in_npy(
    #     out_file=os.path.join(out_dir, os.path.basename(csv_file).replace('.csv', '_clusters.npy')),
    #     data=out_pos
    # )
    # store_point_and_vectors_in_vtp(
    #     out_path=os.path.join(out_dir, os.path.basename(csv_file).replace('.csv', '_clusters.vtp')),
    #     in_points=out_pos,
    #     in_scalars=[out_p_num, np.arange(out_pos.shape[0])],
    # )
    print(out_pos.shape, "clusters found")
    store_array_in_star(
        out_file=os.path.join(out_dir, os.path.basename(csv_file).replace('.csv', '_clusters.star')),
        data=out_pos,
        header=["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"]
    )