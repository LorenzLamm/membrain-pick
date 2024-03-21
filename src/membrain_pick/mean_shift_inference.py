from membrain_seg.segmentation.dataloading.data_utils import get_csv_data, store_array_in_csv, store_point_and_vectors_in_vtp
from membrain_pick.optimization.mean_shift_utils import MeanShiftForwarder
import numpy as np
import torch
import os



def mean_shift_for_scores(
        positions: np.ndarray,
        scores: np.ndarray,
        bandwidth: float,
        max_iter: int,
        margin: float,
        device: str,
        score_threshold: float = 9.0,
):
    

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
    store_array_in_csv(
        out_file=os.path.join(out_dir, os.path.basename(csv_file).replace('.csv', '_clusters.csv')),
        data=out_pos,
    )
    store_point_and_vectors_in_vtp(
        out_path=os.path.join(out_dir, os.path.basename(csv_file).replace('.csv', '_clusters.vtp')),
        in_points=out_pos,
        in_scalars=[out_p_num],
    )