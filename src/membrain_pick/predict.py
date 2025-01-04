import os
from tqdm import tqdm

import torch
import numpy as np

from membrain_pick.dataloading.diffusionnet_datamodule import (
    MemSegDiffusionNetDataModule,
)
from membrain_pick.optimization.diffusion_training_pylit import DiffusionNetModule

from membrain_pick.dataloading.data_utils import (
    store_mesh_in_hdf5,
)
from membrain_pick.clustering.mean_shift_inference import (
    mean_shift_for_scores,
    store_clusters,
)


def save_output(cur_mb_data, out_dir, mb_token):
    out_file_csv = os.path.join(out_dir, f"{mb_token}.csv")

    all_verts = np.concatenate(cur_mb_data["verts"], axis=0)
    all_scores = np.concatenate(cur_mb_data["scores"], axis=0)
    all_labels = np.concatenate(cur_mb_data["labels"], axis=0)
    all_features = np.concatenate(cur_mb_data["features"], axis=0)
    all_weights = np.concatenate(cur_mb_data["weights"], axis=0)
    all_faces = np.concatenate(cur_mb_data["faces"], axis=0)

    # Find unique verts and their inverse indices
    unique_verts, inverse_indices = np.unique(all_verts, axis=0, return_inverse=True)
    inverse_indices = np.squeeze(inverse_indices)

    # Initialize arrays to store the aggregated results
    unique_scores = np.zeros(unique_verts.shape[0])
    unique_features = np.zeros((unique_verts.shape[0], all_features.shape[1]))
    unique_labels = np.zeros(unique_verts.shape[0])

    all_faces = np.array(all_faces, dtype=int)
    new_faces = inverse_indices[all_faces]
    new_faces = np.unique(new_faces, axis=0)

    # Compute weighted average of scores, and select the first labels and features
    for i in range(unique_verts.shape[0]):
        indices = np.where(inverse_indices == i)[0]
        weights = all_weights[indices]
        unique_scores[i] = np.average(all_scores[indices], weights=weights)
        # unique_scores[i] = np.max(all_scores[indices])
        unique_labels[i] = all_labels[indices[0]]
        unique_features[i] = all_features[indices[0]]

    return unique_verts, unique_scores, out_file_csv, unique_labels, new_faces


def save_output_h5(
    unique_verts,
    new_faces,
    unique_scores,
    unique_labels,
    out_file_csv,
    cluster_centers=None,
    tomo_file="",
    pixel_size=10.0,
):
    out_file_h5 = out_file_csv.replace(".csv", ".h5")
    print(f"Saving to {out_file_h5}")
    store_mesh_in_hdf5(
        out_file=out_file_h5,
        points=unique_verts,
        faces=new_faces,
        scores=unique_scores,
        labels=unique_labels,
        cluster_centers=cluster_centers,
        tomo_file=tomo_file,
        pixel_size=pixel_size,
    )


def predict(
    data_dir: str,
    ckpt_path: str,
    out_dir: str,
    is_single_mb: bool = False,
    # Dataset parameters
    partition_size: int = 2000,
    input_pixel_size: float = 10.0,
    force_recompute_partitioning: bool = False,
    k_eig: int = 128,
    N_block: int = 4,
    C_width: int = 64,
    conv_width: int = 32,
    # Mean shift parameters
    mean_shift_output: bool = False,
    mean_shift_bandwidth: float = 7.0,
    mean_shift_max_iter: int = 150,
    mean_shift_margin: float = 0.0,
    mean_shift_score_threshold: float = 9.0,
    # mean_shift_device: str = "cuda:0",
    mean_shift_device: str = "cpu",
):
    """Predict the output of the trained model on the given data.

    Args:
        data_dir (str): The directory containing the data to prespandict.
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
        cache_dir=f"{out_dir}/mb_cache",  # always recompute partitioning
        input_pixel_size=input_pixel_size,
        k_eig=k_eig,
        batch_size=1,
        force_recompute=force_recompute_partitioning,
        num_workers=0,
        pin_memory=False,
        overfit=False,
    )
    data_module.setup(stage="test")
    test_loader = data_module.test_dataloader()
    # test_loader = data_module.train_dataloader()

    model = DiffusionNetModule.load_from_checkpoint(
        ckpt_path,
        map_location=device,
        strict=False,
        dropout=False,
        N_block=N_block,
        C_width=C_width,
        conv_width=conv_width,
        # C_in=10,
        C_in=16,
        one_D_conv_first=True,
    )
    model.to(device)
    model.eval()

    os.makedirs(out_dir, exist_ok=True)

    prev_mb_nr = 0
    verts_count = 0
    prev_mb_token = ""
    prev_tomo_file = ""
    cur_mb_data = {
        "verts": [],
        "faces": [],
        "scores": [],
        "labels": [],
        "features": [],
        "weights": [],
    }

    for i, batch in tqdm(enumerate(test_loader)):
        all_diffusion_feature = batch["diffusion_inputs"]["features"].clone()

        outputs = []
        for i in range(all_diffusion_feature.shape[1] - 15):
            batch["diffusion_inputs"]["features"] = all_diffusion_feature[:, i : i + 16]
            with torch.no_grad():
                output = model(batch)
            outputs.append(output["mse"].squeeze().detach().cpu().numpy())

        def aggregate_outputs(outputs):
            mse_agg = np.stack(outputs, axis=1)
            mse_agg = np.min(mse_agg, axis=1)
            return {"mse": mse_agg}

        output = aggregate_outputs(outputs)
        vert_weights = batch["vert_weights"]
        cur_mb_nr = batch["mb_idx"]
        mb_token = batch["mb_token"]
        tomo_file = batch["tomo_file"]
        if cur_mb_nr != prev_mb_nr:
            verts_count = 0
        faces = batch["faces"] + verts_count
        verts_count += batch["verts_orig"].shape[0]

        if cur_mb_nr != prev_mb_nr:
            unique_verts, unique_scores, out_file_csv, unique_labels, new_faces = (
                save_output(cur_mb_data, out_dir, prev_mb_token)
            )
            clusters = None
            if mean_shift_output:
                print("Performing mean shift...")
                clusters, out_p_num = mean_shift_for_scores(
                    positions=unique_verts,
                    scores=unique_scores,
                    bandwidth=mean_shift_bandwidth,
                    max_iter=mean_shift_max_iter,
                    margin=mean_shift_margin,
                    score_threshold=mean_shift_score_threshold,
                    device=mean_shift_device,
                )

                store_clusters(
                    csv_file=out_file_csv,
                    out_dir=out_dir,
                    out_pos=clusters,
                    out_p_num=out_p_num,
                    verts=unique_verts,
                    faces=new_faces,
                )
            save_output_h5(
                unique_verts,
                new_faces,
                unique_scores,
                unique_labels,
                out_file_csv,
                cluster_centers=clusters,
                tomo_file=prev_tomo_file,
                pixel_size=input_pixel_size,
            )
            cur_mb_data = {
                "verts": [],
                "scores": [],
                "labels": [],
                "features": [],
                "weights": [],
                "faces": [],
            }
            prev_mb_nr = cur_mb_nr
            prev_tomo_file = tomo_file

        prev_mb_token = mb_token
        prev_tomo_file = tomo_file

        cur_mb_data["verts"].append(batch["verts_orig"].detach().cpu().numpy())
        cur_mb_data["scores"].append(output["mse"].squeeze())
        cur_mb_data["labels"].append(batch["label"].detach().cpu().numpy())
        cur_mb_data["features"].append(batch["membrane"][:, 3:].detach().cpu().numpy())
        cur_mb_data["weights"].append(vert_weights.detach().cpu().numpy())
        cur_mb_data["faces"].append(faces.detach().cpu().numpy())

    unique_verts, unique_scores, out_file_csv, unique_labels, new_faces = save_output(
        cur_mb_data, out_dir, mb_token
    )
    clusters = None
    if mean_shift_output:
        print("Performing mean shift...")
        clusters, out_p_num = mean_shift_for_scores(
            positions=unique_verts,
            scores=unique_scores,
            bandwidth=mean_shift_bandwidth,
            max_iter=mean_shift_max_iter,
            margin=mean_shift_margin,
            score_threshold=mean_shift_score_threshold,
            device=mean_shift_device,
        )
        if clusters.shape[0] == 0:
            clusters = np.zeros((0, 3))
        store_clusters(
            csv_file=out_file_csv,
            out_dir=out_dir,
            out_pos=clusters,
            out_p_num=out_p_num,
            verts=unique_verts,
            faces=new_faces,
        )
    save_output_h5(
        unique_verts,
        new_faces,
        unique_scores,
        unique_labels,
        out_file_csv,
        cluster_centers=clusters,
        tomo_file=prev_tomo_file,
        pixel_size=input_pixel_size,
    )
