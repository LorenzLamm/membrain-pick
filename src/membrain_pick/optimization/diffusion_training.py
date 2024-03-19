import numpy as np
import torch
import os

from membrain_pick.networks import diffusion_net
from membrain_pick.optimization.optim_utils import save_checkpoint

import wandb

from membrain_pick.optimization.mean_shift_losses import MeanShift_loss
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R

from membrain_pick.optimization.mean_shift_utils import MeanShiftForwarder

from matplotlib import pyplot as plt
import io
from PIL import Image


def project_and_rotate_points(point_cloud, normal_vector, point_on_plane):
    # Step 1: Project points onto the plane
    # Calculate distance from each point to the plane
    distances = np.dot(point_cloud - point_on_plane, normal_vector)
    projected_points = point_cloud - np.outer(distances, normal_vector)

    # Step 2: Rotate the projected points
    # Calculate the rotation needed to align the normal vector with the z-axis
    z_axis = np.array([0, 0, 1])
    axis_of_rotation = np.cross(normal_vector, z_axis)
    angle_of_rotation = np.arccos(np.dot(normal_vector, z_axis) / np.linalg.norm(normal_vector))
    rotation_vector = axis_of_rotation / np.linalg.norm(axis_of_rotation) * angle_of_rotation
    rotation = R.from_rotvec(rotation_vector)
    rotated_points = rotation.apply(projected_points)

    # Ensure the z-component is zero by subtracting minimal z-value
    rotated_points[:, 2] -= np.min(rotated_points[:, 2])

    return rotated_points


def find_best_fit_plane(point_cloud):
    # Center the point cloud
    point_cloud_mean = np.mean(point_cloud, axis=0)
    centered_point_cloud = point_cloud - point_cloud_mean

    # Perform PCA
    pca = PCA(n_components=3)
    pca.fit(centered_point_cloud)

    # The normal vector to the plane is the eigenvector corresponding to the smallest eigenvalue
    normal_vector = pca.components_[2]

    # The plane is defined by the normal vector and any point on the plane, for example, the mean point
    point_on_plane = point_cloud_mean

    return normal_vector, point_on_plane


def load_data_from_batch(epoch, batch, use_faces, cache_dir, k_eig=128, use_precomputed_normals=True, normalize_verts=True, hks_features=False, augment_random_rotate=False, aggregate_coordinates=False,
                         random_sample_thickness=False, classification=False, distance_radius=7.):
    points = batch["membrane"][0, 0].float()
    labels = batch["label"][0, 0].float()
    mb_idx = batch["mb_idx"]

    if classification:
        
        labels = (labels < distance_radius) * 1.0

    faces = batch["faces"][0, 0]
    gt_pos = batch["gt_pos"][0].float()
    if not use_faces:
        faces = torch.zeros((0, 3))
    normals = batch["normals"][0, 0].float()
    point_weights = batch["vert_weights"][0, 0].float()

    verts = points[:, :3]
    features = points[:, 3:]
    feature_len = 10

    if random_sample_thickness:
        start_sample = np.random.randint(0, features.shape[1] - feature_len)
        features = features[:, start_sample:start_sample + feature_len]
    
    if normalize_verts:
        verts = diffusion_net.geometry.normalize_positions(verts)
    else:
        verts_orig = verts.clone()
        verts = verts.contiguous()
        verts *= 14.08
        verts -= verts.mean()


   
    # Get the geometric operators needed to evaluate DiffusionNet. This routine 
    # automatically populates a cache, precomputing only if needed.
    try:
        frames, mass, L, evals, evecs, gradX, gradY = \
            diffusion_net.geometry.get_operators(verts, faces, op_cache_dir=cache_dir, k_eig=k_eig, normals=(normals if use_precomputed_normals else None), overwrite_cache=(epoch == 0))
    except:
        return None, None, None, None, None, None, None, None, None, None, None, None, None, None
    
    if augment_random_rotate:
        verts = diffusion_net.utils.random_rotate_points(verts)
    
    if hks_features:
        features_hks = diffusion_net.geometry.compute_hks_autoscale(evals, evecs, 16)
        features = torch.cat([features, features_hks], dim=1)
    
    if aggregate_coordinates:
        features = torch.cat([verts, features], dim=1)

    # Convert all inputs to float32 before passing them to the model
    features = features.float()  # You've already done this
    mass = mass.float()
    L = L.float()
    evals = evals.float()
    evecs = evecs.float()
    gradX = gradX.float()
    gradY = gradY.float()
    faces = faces.float()
    return features, mass, L, evals, evecs, gradX, gradY, faces, labels, gt_pos.float(), verts, mb_idx, verts_orig, point_weights

def training_step(epoch,
                  model, 
                  dataloader, 
                  use_faces, 
                  cache_dir, 
                  optimizer, 
                  loss_fn, 
                  mean_losses,
                  mean_losses_ms,
                  k_eig=128,
                  use_precomputed_normals=True,
                  normalize_verts=True,
                  hks_features=False, 
                  augment_random_rotate=False,
                  use_mean_shift=False,
                  aggregate_coordinates=False,
                  random_sample_thickness=False,
                  classification=False,
                  distance_radius=7.,
                  ):
    ms_module = MeanShiftForwarder(
                        bandwidth=7.,
                        num_seeds=100,
                        max_iter=10,
                        margin=2.,
                        device="cuda:0",
                    )
    
    mb_idx_loss_dict = {}
    for sample in dataloader:
            
            idx_dict = sample
            if not "diffusion_inputs" in idx_dict.keys():
                continue
            point_weights = idx_dict["vert_weights"].float().squeeze(0)
            verts_orig = idx_dict["verts_orig"].float()
            gt_pos = idx_dict["gt_pos"].float()
            labels = idx_dict["label"].float()
            mb_idx = idx_dict["mb_idx"]


            model = model.float()  # Ensure the model parameters are float32
            model.train()
            # Zero the gradients before running the backward pass.
            optimizer.zero_grad()
            
            # Forward-evaluate the model
            # preds is a NxC_out array of values
            outputs = model(
                x_in=idx_dict["diffusion_inputs"]["features"],
                mass=idx_dict["diffusion_inputs"]["mass"],
                L=idx_dict["diffusion_inputs"]["L"],
                evals=idx_dict["diffusion_inputs"]["evals"],
                evecs=idx_dict["diffusion_inputs"]["evecs"],
                gradX=idx_dict["diffusion_inputs"]["gradX"],
                gradY=idx_dict["diffusion_inputs"]["gradY"],
                faces=idx_dict["diffusion_inputs"]["faces"],
            )

            # features, mass, L, evals, evecs, gradX, gradY, faces, labels, gt_pos, verts, mb_idx, verts_orig, point_weights = load_data_from_batch(epoch, 
            #                                                                                     sample, 
            #                                                                                     use_faces, 
            #                                                                                     cache_dir, 
            #                                                                                     k_eig=k_eig, 
            #                                                                                     use_precomputed_normals=use_precomputed_normals, 
            #                                                                                     normalize_verts=normalize_verts, 
            #                                                                                     hks_features=hks_features,
            #                                                                                     augment_random_rotate=augment_random_rotate,
            #                                                                                     aggregate_coordinates=aggregate_coordinates,
            #                                                                                     random_sample_thickness=random_sample_thickness,
            #                                                                                     classification=classification,
            #                                                                                     distance_radius=distance_radius,
            #                                                                                     )
            # if features is None:
            #     continue
            
            # model = model.float()  # Ensure the model parameters are float32
            # model.train()
            # # Zero the gradients before running the backward pass.
            # optimizer.zero_grad()
            
            # Forward-evaluate the model
            # preds is a NxC_out array of values
            # outputs = model(features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces, verts=verts)
            ms_loss = 0.
            if use_mean_shift:
                if gt_pos.shape[0] != 0 and torch.sum(labels < 10.) > 10:
                    
                    close_mask = labels < 10.
                    ms_verts = verts_orig[close_mask] * 928
                    ms_features = outputs[close_mask]
                    outputs_ms, _, _ = ms_module.mean_shift_forward(
                            x=ms_verts.squeeze().clone(),
                            weights=ms_features.squeeze(),
                        )
                    ms_loss = MeanShift_loss(True)
                    ms_loss, _, _ = ms_loss(gt_pos.to(model.device) * 928, outputs_ms.to(model.device), print_flag=False)

                # outputs_ms = outputs[1]
                # outputs = outputs[0]
                # if gt_pos.shape[0] != 0:


            outputs = outputs.squeeze().float()
            # Compute and print loss
            
            # print(outputs.shape, labels.shape, "SHAPE")

            # multiply by point weights to get the weighted loss
            outputs *= point_weights
            labels *= point_weights

            loss = loss_fn(outputs, labels.squeeze()).float()
            # if use_mean_shift:
            #     loss *= 0.

            if use_mean_shift and gt_pos.shape[0] != 0:
                if ms_loss != 0.:
                    loss += ms_loss.to(loss.device) 
            # print(loss, "LOSS")
            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # Update parameters
            optimizer.step()
            mean_losses.append(loss.item())
            mean_losses_ms.append(ms_loss if isinstance(ms_loss, float) else ms_loss.item()) 
            if int(mb_idx) in mb_idx_loss_dict:
                mb_idx_loss_dict[int(mb_idx)][0] += loss.item()
                mb_idx_loss_dict[int(mb_idx)][1] += 1
            else:
                mb_idx_loss_dict[int(mb_idx)] = [loss.item(), 1]
    for key in mb_idx_loss_dict:
        mb_idx_loss = mb_idx_loss_dict[key][0] / mb_idx_loss_dict[key][1]
        print(mb_idx_loss, "LOSS for MB IDX TRAIN", key, flush=True)
        # print("This consists of number of samples", mb_idx_loss_dict[key][1], flush=True)

    return model, optimizer, mean_losses, mean_losses_ms
     

def validation_step(epoch,
                    model, 
                    dataloader, 
                    use_faces, 
                    cache_dir, 
                    loss_fn, 
                    k_eig=128,
                    use_precomputed_normals=True,
                    normalize_verts=True,
                    hks_features=False,
                    use_mean_shift=False,
                    aggregate_coordinates=False,
                    random_sample_thickness=False,
                    classification=False,
                    distance_radius=7.,
                    plot_projection=False,
                   ):
    
    ms_module = MeanShiftForwarder(
                        bandwidth=7.,
                        num_seeds=100,
                        max_iter=10,
                        margin=2.,
                        device="cuda:0",
                    )
    
    mean_losses = []
    mean_losses_ms = []
    model = model.float()  # Ensure the model parameters are float32
    model.eval()  # Set the model to evaluation mode

    plot_projection = True
    if epoch % 10 == 0 and plot_projection:
        plot_flag = True
    else:
        plot_flag = False
    prev_mb_idx = None

    sample_preds = []
    sample_verts = []
    sample_gt = []
    sample_gt_pos = []
    sample_pred_pos_ms = []
    sample_features = []
    store_img_container = None
    mb_idx_loss_dict = {}

    with torch.no_grad():  # No gradients needed for validation
        for sample_id, sample in enumerate(dataloader):

            idx_dict = sample
            if not "diffusion_inputs" in idx_dict.keys():
                continue

            point_weights = idx_dict["vert_weights"].float().squeeze()
            verts_orig = idx_dict["verts_orig"].float()
            gt_pos = idx_dict["gt_pos"].float()
            labels = idx_dict["label"].float()
            mb_idx = idx_dict["mb_idx"]

            outputs = model(
                x_in=idx_dict["diffusion_inputs"]["features"],
                mass=idx_dict["diffusion_inputs"]["mass"],
                L=idx_dict["diffusion_inputs"]["L"],
                evals=idx_dict["diffusion_inputs"]["evals"],
                evecs=idx_dict["diffusion_inputs"]["evecs"],
                gradX=idx_dict["diffusion_inputs"]["gradX"],
                gradY=idx_dict["diffusion_inputs"]["gradY"],
                faces=idx_dict["diffusion_inputs"]["faces"],
            )
            # features, mass, L, evals, evecs, gradX, gradY, faces, labels, gt_pos, verts, mb_idx, verts_orig, point_weights = load_data_from_batch(epoch,
            #                                                                                     sample, 
            #                                                                                     use_faces, 
            #                                                                                     cache_dir, 
            #                                                                                     k_eig=k_eig, 
            #                                                                                     use_precomputed_normals=use_precomputed_normals, 
            #                                                                                     normalize_verts=normalize_verts, 
            #                                                                                     hks_features=hks_features,
            #                                                                                     aggregate_coordinates=aggregate_coordinates,
            #                                                                                     random_sample_thickness=random_sample_thickness,
            #                                                                                     classification=classification,
            #                                                                                     distance_radius=distance_radius,
            #                                                                                     )
            
            # if features is None:
            #     continue

            # # Forward-evaluate the model
            # outputs = model(features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces).squeeze().float()

            # # Compute loss
            # loss = loss_fn(outputs, labels).float()

             # preds is a NxC_out array of values
            
            # outputs = model(features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces, verts=verts)
            ms_loss = 0.
            if use_mean_shift:
                if gt_pos.shape[0] != 0 and torch.sum(labels < 10.) > 10:
                    
                    close_mask = labels < 10.
                    ms_verts = verts_orig[close_mask] * 928
                    ms_features = outputs[close_mask]
                    outputs_ms, _, _ = ms_module.mean_shift_forward(
                            x=ms_verts.squeeze().clone(),
                            weights=ms_features.squeeze(),
                        )
                    ms_loss = MeanShift_loss(True)
                    ms_loss, _, _ = ms_loss(gt_pos.to(model.device) * 928, outputs_ms.to(model.device), print_flag=False)

            # outputs = model(features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces, verts=verts)

            # if use_mean_shift:
            #     outputs_ms = outputs[1]
            #     outputs = outputs[0]
                
            #     if gt_pos.shape[0] != 0:
            #         ms_loss = MeanShift_loss(True)
            #         ms_loss, _, _ = ms_loss(gt_pos.to(model.device), outputs_ms.to(model.device))

            outputs = outputs.squeeze().float()

            outputs *= point_weights
            labels *= point_weights
            # Compute and print loss
            loss = loss_fn(outputs, labels.squeeze()).float()


            if plot_flag and mb_idx != prev_mb_idx or sample_id == len(dataloader) - 1:
                store_img_container = {}
                if prev_mb_idx is None:
                    prev_mb_idx = mb_idx
                else:
                    prev_mb_idx = mb_idx
                    sample_preds = np.concatenate(sample_preds, axis=0)
                    sample_gt = np.concatenate(sample_gt, axis=0)
                    sample_gt_pos = np.concatenate(sample_gt_pos, axis=0)
                    sample_verts = np.concatenate(sample_verts, axis=0)
                    sample_features = np.concatenate(sample_features, axis=0)
                    if use_mean_shift:
                        sample_pred_pos_ms = np.concatenate(sample_pred_pos_ms, axis=0)

                    # remove duplicates
                    unique_verts = np.unique(sample_verts, axis=0)
                    unique_gt_pos = np.unique(sample_gt_pos, axis=0)
                    unique_scores = np.zeros((unique_verts.shape[0]))
                    unique_labels = np.zeros((unique_verts.shape[0]))
                    unique_features = np.zeros((unique_verts.shape[0], sample_features.shape[1]))

                    if use_mean_shift:
                        unique_pred_pos_ms = np.unique(sample_pred_pos_ms, axis=0)
                    
                    for i, vert in enumerate(unique_verts):
                        idxs = np.where(np.all(sample_verts == vert, axis=1))[0]
                        unique_scores[i] = np.mean(sample_preds[idxs], axis=0)
                        unique_labels[i] = sample_gt[idxs[0]]
                        unique_features[i] = sample_features[idxs[0]]
                    
                    if plot_flag:
                        normal_vector, point_in_plane = find_best_fit_plane(unique_verts)
                        projected_points = project_and_rotate_points(unique_verts, normal_vector, point_in_plane)
                        projected_points = projected_points[:, :2]
                        projected_gt = project_and_rotate_points(unique_gt_pos, normal_vector, point_in_plane)
                        if use_mean_shift:
                            
                            projected_pred_pos_ms = project_and_rotate_points(unique_pred_pos_ms / 928., normal_vector, point_in_plane)
                            projected_pred_pos_ms = projected_pred_pos_ms[:, :2]
                            store_img_container["projected_pred_pos_ms"] = projected_pred_pos_ms
                        projected_gt = projected_gt[:, :2]
                        mask = unique_labels <= 10.
                        projected_points = projected_points[mask]
                        unique_labels = unique_labels[mask]
                        unique_features = unique_features[mask]
                        unique_scores = unique_scores[mask]

                        store_img_container["projected_points"] = projected_points
                        store_img_container["projected_gt"] = projected_gt
                        store_img_container["unique_labels"] = unique_labels
                        store_img_container["unique_scores"] = unique_scores
                        store_img_container["unique_features"] = unique_features

                        # print(projected_gt, "<-- gt")
                        # print("---")
                        # print(projected_pred_pos_ms, "<-- pred")

                        # from matplotlib import pyplot as plt
                        # plt.figure()
                        # plt.scatter(projected_points[:, 0], projected_points[:, 1], c=unique_labels, s=1.5)
                        # plt.clim(0, 10)
                        # plt.scatter(projected_gt[:, 0], projected_gt[:, 1], c="r", s=3.0)
                        # plt.colorbar()
                        # plt.savefig("test_gt.png")
                        # plt.figure()
                        # plt.scatter(projected_points[:, 0], projected_points[:, 1], c=unique_scores, s=1.5)
                        # plt.scatter(projected_gt[:, 0], projected_gt[:, 1], c="r", s=3.0)
                        # plt.clim(0, 10)
                        # plt.colorbar()
                        # plt.savefig("test_preds.png")
                    sample_preds = []
                    sample_verts = []
                    sample_gt = []
                    sample_gt_pos = []
                    sample_pred_pos_ms = []
                    sample_features = []

            features = idx_dict["diffusion_inputs"]["features"]
            if plot_flag:
                sample_preds.append(outputs.cpu().detach().numpy())
                sample_gt.append(labels.cpu().detach().numpy().squeeze(0))
                sample_gt_pos.append(gt_pos.cpu().detach().numpy().squeeze(0))
                sample_verts.append(verts_orig.cpu().detach().numpy().squeeze(0))
                sample_features.append(features.cpu().detach().numpy().squeeze(0))
                if use_mean_shift:
                    sample_pred_pos_ms.append(outputs_ms.cpu().detach().numpy())

            

            # if use_mean_shift:
            #     loss *= 0.
            if use_mean_shift and gt_pos.shape[0] != 0:
                if ms_loss != 0.:
                    loss = loss + ms_loss.to(loss.device) 
            mean_losses.append(loss.item())
            mean_losses_ms.append(ms_loss if isinstance(ms_loss, float) else ms_loss.item())
            if int(mb_idx) in mb_idx_loss_dict:
                mb_idx_loss_dict[int(mb_idx)][0] += loss.item()
                mb_idx_loss_dict[int(mb_idx)][1] += 1
            else:
                mb_idx_loss_dict[int(mb_idx)] = [loss.item(), 1]

    for key in mb_idx_loss_dict:
        mb_idx_loss = mb_idx_loss_dict[key][0] / mb_idx_loss_dict[key][1]
        print(mb_idx_loss, "LOSS for MB IDX", key, flush=True)
        # print("This consists of number of samples", mb_idx_loss_dict[key][1], flush=True)
    # Calculate and return the mean loss over all batches in the dataloader
    mean_loss = np.mean(mean_losses)
    mean_loss_ms = np.mean(mean_losses_ms)
    return mean_loss, mean_loss_ms, store_img_container


def training_routine(model,
                     run_token, 
                     out_folder, 
                     cache_dir, 
                     train_loader, 
                     val_loader,
                     loss_fn, 
                     optimizer, 
                     scheduler, 
                     hks_features=False,
                     use_faces=True, 
                     use_precomputed_normals=True,
                     num_epochs=10000,
                     log_every_nth=30,
                     validate_every_nth_epoch=10,
                     normalize_verts=True,
                     augment_random_rotate=False,
                     use_mean_shift=False,
                     wandb_logging=True,
                     aggregate_coordinates=False,
                     random_sample_thickness=False,
                     classification=False,
                     distance_radius=7.,
                     ):
    

    losses = []
    mean_losses_train = []
    mean_losses_val = []
    cur_best_val_loss = 100000
    cur_best_train_loss = 100000
    cur_checkpoint_file = run_token + '_checkpoint_epoch-1'
    current_best_epoch_val = -1
    current_best_epoch_train = -1
    cur_training_ckpt = "dummy"

    plot_img1 = None
    plot_img2 = None
    
    for epoch in range(num_epochs):
        scheduler.step(cur_best_train_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print("EPOCH", epoch + 1, "  ...   Mean Loss:", np.mean(mean_losses_train), flush=True)
        losses.append(np.mean(mean_losses_train))
        if np.mean(mean_losses_train) < cur_best_train_loss:
            print("Saving checkpoint")
            cur_best_train_loss = np.mean(mean_losses_train)
            if os.path.isfile(cur_training_ckpt):
                os.remove(cur_training_ckpt)
            cur_training_ckpt = save_checkpoint(model, optimizer, epoch, cur_best_train_loss, checkpoint_dir=out_folder, filename=run_token + '_checkpoint_epoch%d' % epoch)
            # remove old checkpoints
                    
                    
        mean_losses_train = []
        mean_losses_train_ms = []
        model, optimizer, mean_loss, mean_loss_ms = training_step(epoch, 
                                                    model,
                                                      train_loader,
                                                        use_faces,
                                                        cache_dir,
                                                        optimizer,
                                                        loss_fn,
                                                        mean_losses_train,
                                                        mean_losses_train_ms,
                                                        use_precomputed_normals=use_precomputed_normals,
                                                        hks_features=hks_features,
                                                        normalize_verts=normalize_verts,
                                                        augment_random_rotate=augment_random_rotate,
                                                        use_mean_shift=use_mean_shift,
                                                        aggregate_coordinates=aggregate_coordinates,
                                                        random_sample_thickness=random_sample_thickness,
                                                        classification=classification,
                                                        distance_radius=distance_radius,
                                                        )
        
        # mean_losses_train.append(np.mean(mean_loss))
        mean_losses_train = [np.mean(mean_loss)]
        mean_losses_ms = [np.mean(mean_loss_ms)]

        if epoch % validate_every_nth_epoch == 0:
            mean_loss_val, mean_loss_ms_val, store_img_container = validation_step(epoch,
                                            model,
                                            val_loader,
                                            use_faces,
                                            cache_dir,
                                            loss_fn,
                                            hks_features=hks_features,
                                            normalize_verts=normalize_verts,
                                            use_precomputed_normals=use_precomputed_normals,
                                            use_mean_shift=use_mean_shift,
                                            aggregate_coordinates=aggregate_coordinates,
                                            random_sample_thickness=random_sample_thickness,
                                            classification=classification,
                                            distance_radius=distance_radius,
                                            )
            mean_losses_val.append(mean_loss_val)
            plot_img1, plot_img2 = None, None
            if wandb_logging and store_img_container is not None and "projected_points" in store_img_container:
                print("Logging image")
                plot_img1, plot_img2 = log_image(store_img_container)

            print("Validation Loss:", mean_loss_val)
            if mean_loss_val < cur_best_val_loss:
                print("Saving checkpoint")
                cur_best_val_loss = mean_loss_val
                # remove old checkpoints
                if os.path.isfile(cur_checkpoint_file):
                    os.remove(cur_checkpoint_file)
                cur_checkpoint_file = run_token + '_checkpoint_epoch%d' % epoch + "_val" + str(mean_loss_val)
                cur_checkpoint_file = save_checkpoint(model, optimizer, epoch, cur_best_val_loss, checkpoint_dir=out_folder, filename=cur_checkpoint_file)
                
            mean_losses_val = []

        # Log training and validation losses and learning rate
        wandb.log({"Epoch": epoch, 
                   "Training Loss": np.mean(mean_losses_train), 
                   "Training MS Loss": np.mean(mean_losses_ms), 
                   "Validation Loss": mean_loss_val,
                   "Learning Rate": current_lr, 
                   "Validation MS Loss": mean_loss_ms_val, })
        if plot_img1 is not None and plot_img2 is not None:
            wandb.log({"Ground truth": (wandb.Image(plot_img1) if plot_img1 is not None else "No image"),
                       "Prediction": (wandb.Image(plot_img2) if plot_img2 is not None else "No image"),})
            

            


def plot_and_log_to_wandb(projected_points, projected_gt, unique_labels, unique_scores, title, pred_pos=None):
    plt.figure()
    scatter = plt.scatter(projected_points[:, 0], projected_points[:, 1], c=unique_labels if title == "Ground Truth" else unique_scores, s=1.5)
    if pred_pos is not None:
        plt.scatter(pred_pos[:, 0], pred_pos[:, 1], c="g", s=2.0)
    plt.scatter(projected_gt[:, 0], projected_gt[:, 1], c="r", s=3.0)
    plt.clim(0, 10)
    plt.colorbar()
    plt.title(title)

    # Save the plot to a BytesIO buffer instead of a file
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)  # Go to the beginning of the IO buffer

    # Create a PIL Image from the buffer
    image = Image.open(buf)

    # Log the plot to wandb
    # wandb.log({f"{title} Plot": wandb.Image(image)})
    plt.close()  # Close the plot to free memory
    return image

def log_image(image_container):
    """ Create an image from the container contents and log it to wandb """
    img1 = plot_and_log_to_wandb(image_container["projected_points"], image_container["projected_gt"], image_container["unique_labels"], image_container["unique_scores"], "Ground Truth")
    if "projected_pred_pos_ms" in image_container:
        img2 = plot_and_log_to_wandb(image_container["projected_points"], image_container["projected_gt"], image_container["unique_labels"], image_container["unique_scores"], "Predictions", pred_pos=image_container["projected_pred_pos_ms"])
    else:
        img2 = plot_and_log_to_wandb(image_container["projected_points"], image_container["projected_gt"], image_container["unique_labels"], image_container["unique_scores"], "Predictions")
    return img1, img2

