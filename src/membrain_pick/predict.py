import os 
from tqdm import tqdm

import torch
import numpy as np
import pytorch_lightning as pl

from membrain_pick.dataloading.diffusionnet_datamodule import MemSegDiffusionNetDataModule
from membrain_pick.optimization.diffusion_training_pylit import DiffusionNetModule

from membrain_pick.dataloading.data_utils import store_array_in_csv, store_point_and_vectors_in_vtp
from membrain_pick.mean_shift_inference import mean_shift_for_scores, store_clusters


def save_output(cur_mb_data, out_dir, mb_token):
    out_file_csv = os.path.join(out_dir, f"{mb_token}.csv")
    out_file_vtp = os.path.join(out_dir, f"{mb_token}.vtp")

    all_verts = np.concatenate(cur_mb_data["verts"], axis=0)
    all_scores = np.concatenate(cur_mb_data["scores"], axis=0)
    all_labels = np.concatenate(cur_mb_data["labels"], axis=0)
    all_features = np.concatenate(cur_mb_data["features"], axis=0)
    all_weights = np.concatenate(cur_mb_data["weights"], axis=0)

    # Find unique verts and their inverse indices
    unique_verts, inverse_indices = np.unique(all_verts, axis=0, return_inverse=True)
    
    # Initialize arrays to store the aggregated results
    unique_scores = np.zeros(unique_verts.shape[0])
    unique_features = np.zeros((unique_verts.shape[0], all_features.shape[1]))
    unique_labels = np.zeros(unique_verts.shape[0])

    # Compute weighted average of scores, and select the first labels and features
    for i in range(unique_verts.shape[0]):
        indices = np.where(inverse_indices == i)[0]
        weights = all_weights[indices]
        unique_scores[i] = np.average(all_scores[indices], weights=weights)
        unique_labels[i] = all_labels[indices[0]]
        unique_features[i] = all_features[indices[0]]

    store_array_in_csv(out_file_csv, np.concatenate((unique_verts, np.expand_dims(unique_scores, axis=1)), axis=1))
    store_point_and_vectors_in_vtp(out_file_vtp, unique_verts, in_scalars=[unique_labels, unique_scores] + [unique_features[:, i] for i in range(0, unique_features.shape[1])])

    return unique_verts, unique_scores, out_file_csv


def predict(
        data_dir: str,
        ckpt_path: str,
        out_dir: str,
        is_single_mb: bool = False,

        # Dataset parameters
        partition_size: int = 2000,
        pixel_size: float = 1.0,
        max_tomo_shape: int = 928,
        k_eig: int = 128,

        # Mean shift parameters
        mean_shift_output: bool = False,
        mean_shift_bandwidth: float = 7.,
        mean_shift_max_iter: int = 150,
        mean_shift_margin: float = 0.,
        mean_shift_score_threshold: float = 9.0,
        mean_shift_device: str = "cuda:0",

):
    """Predict the output of the trained model on the given data.

    Args:
        data_dir (str): The directory containing the data to predict.
        out_dir (str): The directory to save the output to.
        predict_entire_dir (bool): Whether to predict the entire directory.
    """

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    data_module = MemSegDiffusionNetDataModule(
        csv_folder_train=None,
        csv_folder_val=None,
        csv_folder_test=data_dir,
        is_single_mb=is_single_mb,
        load_n_sampled_points=partition_size,
        cache_dir="./mb_cache", # always recompute partitioning
        pixel_size=pixel_size,
        max_tomo_shape=max_tomo_shape,
        k_eig=k_eig,
        batch_size=1,
        force_recompute=True,
        num_workers=0,
        pin_memory=False,
        allpos=True
    )
    data_module.setup(stage="test")
    test_loader = data_module.test_dataloader()
    # test_loader = data_module.train_dataloader()

    model = DiffusionNetModule.load_from_checkpoint(ckpt_path,
                                                    map_location=device,
                                                    strict=False,
                                                    dropout=False,
                                                    N_block=4, 
                                                    )
    model.to(device)
    model.eval()

    os.makedirs(out_dir, exist_ok=True)

    prev_mb_nr = 0
    prev_mb_token = ""
    cur_mb_data = {
        "verts": [],
        "scores": [],
        "labels": [],
        "features": [],
        "weights": [],
    }

    for i, batch in tqdm(enumerate(test_loader)):
        with torch.no_grad():
            output = model(batch)
        vert_weights = batch["vert_weights"]
        cur_mb_nr = batch["mb_idx"]
        mb_token = batch["mb_token"]
        labels = batch["label"]
        def mse_loss(output, labels, weights):
            return ((output.cpu().detach()*weights - labels*weights) ** 2).mean()
        loss = mse_loss(output["mse"].squeeze(), labels, weights=vert_weights)
        print(f"Loss: {loss}")
        print("Span:", torch.min(output["mse"].squeeze()), torch.max(output["mse"].squeeze()))

        if cur_mb_nr != prev_mb_nr:
            # if prev_mb_nr != 0: <<---------- I don't think this is necessary / causes issues
            unique_verts, unique_scores, out_file_csv = save_output(cur_mb_data, out_dir, prev_mb_token)
            if mean_shift_output:
                print("Performing mean shift...")
                clusters, out_p_num = mean_shift_for_scores(positions=unique_verts, 
                                                    scores=unique_scores, 
                                                    bandwidth=mean_shift_bandwidth, 
                                                    max_iter=mean_shift_max_iter, 
                                                    margin=mean_shift_margin, 
                                                    score_threshold=mean_shift_score_threshold, 
                                                    device=mean_shift_device)
                store_clusters(
                    csv_file=out_file_csv,
                    out_dir=out_dir,
                    out_pos=clusters,
                    out_p_num=out_p_num,
                )
            cur_mb_data = {
                "verts": [],
                "scores": [],
                "labels": [],
                "features": [],
                "weights": [],
            }
            prev_mb_nr = cur_mb_nr
        
        prev_mb_token = mb_token
        
        cur_mb_data["verts"].append(batch["verts_orig"].detach().cpu().numpy())
        cur_mb_data["scores"].append(output["mse"].squeeze().detach().cpu().numpy())
        cur_mb_data["labels"].append(batch["label"].detach().cpu().numpy())
        cur_mb_data["features"].append(batch["membrane"][:, 3:].detach().cpu().numpy())
        cur_mb_data["weights"].append(vert_weights.detach().cpu().numpy())


    unique_verts, unique_scores, out_file_csv = save_output(cur_mb_data, out_dir, mb_token)
    if mean_shift_output:
        print("Performing mean shift...")
        clusters, out_p_num = mean_shift_for_scores(positions=unique_verts, 
                                         scores=unique_scores, 
                                         bandwidth=mean_shift_bandwidth, 
                                         max_iter=mean_shift_max_iter, 
                                         margin=mean_shift_margin, 
                                         score_threshold=mean_shift_score_threshold, 
                                         device=mean_shift_device)
        store_clusters(
            csv_file=out_file_csv,
            out_dir=out_dir,
            out_pos=clusters,
            out_p_num=out_p_num,
        )


def main():
    data_dir = "/scicore/home/engel0006/GROUP/pool-engel/Lorenz/2D_projections/mesh_data/val"
    data_dir = "/scicore/home/engel0006/GROUP/pool-engel/Lorenz/MemBrain-pick/data/Spinach/meshes/Tomo0001/"
    ckpt_path = "/scicore/home/engel0006/GROUP/pool-engel/Lorenz/MemBrain-pick/scripts/checkpoints/test_diffusion_0-epoch=999-val_loss=1.22.ckpt"
    ckpt_path = "/scicore/home/engel0006/GROUP/pool-engel/Lorenz/MemBrain-pick/scripts/checkpoints/tomo17_run1-epoch=11-val_loss=0.83.ckpt"
    out_dir = "/scicore/home/engel0006/GROUP/pool-engel/Lorenz/MemBrain-pick/evaluations/Spinach/predictions/Tomo0010"
    predict(data_dir, ckpt_path, out_dir, partition_size=2000, pixel_size=15., mean_shift_output=True, mean_shift_bandwidth=5*15.)

if __name__ == "__main__":
    main()