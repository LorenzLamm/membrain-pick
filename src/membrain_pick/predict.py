import os 
from tqdm import tqdm

import torch
import numpy as np
import pytorch_lightning as pl

from membrain_pick.dataloading.diffusionnet_datamodule import MemSegDiffusionNetDataModule
from membrain_pick.optimization.diffusion_training_pylit import DiffusionNetModule

from membrain_pick.dataloading.data_utils import store_array_in_csv, store_point_and_vectors_in_vtp


def save_output(cur_mb_data, out_dir, mb_token):
    out_file_csv = os.path.join(out_dir, f"{mb_token}.csv")
    out_file_vtp = os.path.join(out_dir, f"{mb_token}.vtp")

    all_verts = np.concatenate(cur_mb_data["verts"], axis=0)
    all_scores = np.concatenate(cur_mb_data["scores"], axis=0)
    all_labels = np.concatenate(cur_mb_data["labels"], axis=0)
    all_features = np.concatenate(cur_mb_data["features"], axis=0)
    all_weights = np.concatenate(cur_mb_data["weights"], axis=0)

    # Average scores of duplicates
    print("Finding unique verts...")
    unique_verts = np.unique(all_verts, axis=0)
    unique_scores = np.zeros((unique_verts.shape[0]))
    unique_labels = np.zeros((unique_verts.shape[0]))
    unique_features = np.zeros((unique_verts.shape[0], all_features.shape[1]))
    for i, vert in enumerate(unique_verts):
        idxs = np.where(np.all(all_verts == vert, axis=1))[0]
        weights = all_weights[idxs]
        unique_scores[i] = np.average(all_scores[idxs], axis=0, weights=weights)
        # unique_scores[i] = np.mean(all_scores[idxs], axis=0)
        unique_labels[i] = all_labels[idxs[0]]
        unique_features[i] = all_features[idxs[0]]
    store_array_in_csv(out_file_csv, np.concatenate((unique_verts, np.expand_dims(unique_scores, axis=1)), axis=1))
    store_point_and_vectors_in_vtp(out_file_vtp, unique_verts, in_scalars=[unique_labels, unique_scores] + [unique_features[:, i] for i in range(0, unique_features.shape[1])])


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
        cache_dir=None, # always recompute partitioning
        pixel_size=pixel_size,
        max_tomo_shape=max_tomo_shape,
        k_eig=k_eig,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
    )
    data_module.setup(stage="test")
    test_loader = data_module.test_dataloader()

    model = DiffusionNetModule.load_from_checkpoint(ckpt_path,
                                                    map_location=device,
                                                    strict=False)
    model.to(device)
    model.eval()

    os.makedirs(out_dir, exist_ok=True)

    prev_mb_nr = 0
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
        if cur_mb_nr != prev_mb_nr:
            if prev_mb_nr != 0:
                save_output(cur_mb_data, out_dir, mb_token)
                cur_mb_data = {
                    "verts": [],
                    "scores": [],
                    "labels": [],
                    "features": [],
                    "weights": [],
                }
            prev_mb_nr = cur_mb_nr
        
        cur_mb_data["verts"].append(batch["verts_orig"].detach().cpu().numpy())
        cur_mb_data["scores"].append(output["mse"].squeeze().detach().cpu().numpy())
        cur_mb_data["labels"].append(batch["label"].detach().cpu().numpy())
        cur_mb_data["features"].append(batch["membrane"][:, 3:].detach().cpu().numpy())
        cur_mb_data["weights"].append(vert_weights.detach().cpu().numpy())


    save_output(cur_mb_data, out_dir, mb_token)
