import numpy as np
import torch
import os

from membrain_pick.networks import diffusion_net
from membrain_pick.optimization.optim_utils import save_checkpoint

import wandb

from membrain_pick.optimization.mean_shift_losses import MeanShift_loss


def load_data_from_batch(epoch, batch, use_faces, cache_dir, k_eig=128, use_precomputed_normals=True, normalize_verts=True, hks_features=False, augment_random_rotate=False, aggregate_coordinates=False,
                         random_sample_thickness=False, classification=False, distance_radius=7.):
    points = batch["membrane"][0, 0].float()
    labels = batch["label"][0, 0].float()

    if classification:
        
        labels = (labels < distance_radius) * 1.0

    faces = batch["faces"][0, 0]
    gt_pos = batch["gt_pos"][0].float()
    if not use_faces:
        faces = torch.zeros((0, 3))
    normals = batch["normals"][0, 0].float()

    verts = points[:, :3]
    features = points[:, 3:]
    feature_len = 10

    if random_sample_thickness:
        start_sample = np.random.randint(0, features.shape[1] - feature_len)
        features = features[:, start_sample:start_sample + feature_len]

    if normalize_verts:
        verts = diffusion_net.geometry.normalize_positions(verts)
    else:
        verts = verts.contiguous()
        verts *= 14.08
        verts -= verts.mean()

   
    # Get the geometric operators needed to evaluate DiffusionNet. This routine 
    # automatically populates a cache, precomputing only if needed.
    try:
        frames, mass, L, evals, evecs, gradX, gradY = \
            diffusion_net.geometry.get_operators(verts, faces, op_cache_dir=cache_dir, k_eig=k_eig, normals=(normals if use_precomputed_normals else None), overwrite_cache=(epoch == 0))
    except:
        return None, None, None, None, None, None, None, None, None, None
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
    return features, mass, L, evals, evecs, gradX, gradY, faces, labels, gt_pos.float()

def training_step(epoch,
                  model, 
                  dataloader, 
                  use_faces, 
                  cache_dir, 
                  optimizer, 
                  loss_fn, 
                  mean_losses,
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
    for sample in dataloader:
            features, mass, L, evals, evecs, gradX, gradY, faces, labels, gt_pos = load_data_from_batch(epoch, 
                                                                                                sample, 
                                                                                                use_faces, 
                                                                                                cache_dir, 
                                                                                                k_eig=k_eig, 
                                                                                                use_precomputed_normals=use_precomputed_normals, 
                                                                                                normalize_verts=normalize_verts, 
                                                                                                hks_features=hks_features,
                                                                                                augment_random_rotate=augment_random_rotate,
                                                                                                aggregate_coordinates=aggregate_coordinates,
                                                                                                random_sample_thickness=random_sample_thickness,
                                                                                                classification=classification,
                                                                                                distance_radius=distance_radius,
                                                                                                )
            if features is None:
                continue
            
            model = model.float()  # Ensure the model parameters are float32
            model.train()
            # Zero the gradients before running the backward pass.
            optimizer.zero_grad()
            
            # Forward-evaluate the model
            # preds is a NxC_out array of values
            outputs = model(features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)
            if use_mean_shift:
                outputs_ms = outputs[1]
                outputs = outputs[0]
                if gt_pos.shape[0] != 0:
                    ms_loss = MeanShift_loss(True)
                    print(gt_pos.shape, outputs_ms.shape, "GT POS SHAPE")
                    ms_loss, _, _ = ms_loss(gt_pos.to(model.device), outputs_ms.to(model.device))


            outputs = outputs.squeeze().float()
            # Compute and print loss
            
            # print(outputs.shape, labels.shape, "SHAPE")
            loss = loss_fn(outputs, labels).float()
            if use_mean_shift:
                loss *= 0.

            if use_mean_shift and gt_pos.shape[0] != 0:
                loss += ms_loss.to(loss.device) * 10.
            # print(loss, "LOSS")
            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # Update parameters
            optimizer.step()
            mean_losses.append(loss.item())

    return model, optimizer, mean_losses
     

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
                   ):
    mean_losses = []
    model = model.float()  # Ensure the model parameters are float32
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():  # No gradients needed for validation
        for sample in dataloader:
            features, mass, L, evals, evecs, gradX, gradY, faces, labels, gt_pos = load_data_from_batch(epoch,
                                                                                                sample, 
                                                                                                use_faces, 
                                                                                                cache_dir, 
                                                                                                k_eig=k_eig, 
                                                                                                use_precomputed_normals=use_precomputed_normals, 
                                                                                                normalize_verts=normalize_verts, 
                                                                                                hks_features=hks_features,
                                                                                                aggregate_coordinates=aggregate_coordinates,
                                                                                                random_sample_thickness=random_sample_thickness,
                                                                                                classification=classification,
                                                                                                distance_radius=distance_radius,
                                                                                                )

            # # Forward-evaluate the model
            # outputs = model(features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces).squeeze().float()

            # # Compute loss
            # loss = loss_fn(outputs, labels).float()

             # preds is a NxC_out array of values
            outputs = model(features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)

            if use_mean_shift:
                outputs_ms = outputs[1]
                outputs = outputs[0]
                
                if gt_pos.shape[0] != 0:
                    ms_loss = MeanShift_loss(True)
                    ms_loss, _, _ = ms_loss(gt_pos.to(model.device), outputs_ms.to(model.device))

            outputs = outputs.squeeze().float()
            # Compute and print loss
            loss = loss_fn(outputs, labels).float()

            if use_mean_shift:
                loss *= 0.
            if use_mean_shift and gt_pos.shape[0] != 0:
                loss += ms_loss.to(loss.device) * 10.
            mean_losses.append(loss.item())

    # Calculate and return the mean loss over all batches in the dataloader
    mean_loss = np.mean(mean_losses)
    return mean_loss


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
    
    if wandb_logging:
        # Initialize a new WandB run
        wandb.init(project=run_token, entity="loro")

        wandb.config = {
        "learning_rate": optimizer.param_groups[0]['lr'],
        "epochs": num_epochs,
        "batch_size": 1
        }

    losses = []
    mean_losses_train = []
    mean_losses_val = []
    cur_best_val_loss = 100000
    cur_best_train_loss = 100000
    cur_checkpoint_file = run_token + '_checkpoint_epoch-1'
    current_best_epoch_val = -1
    current_best_epoch_train = -1
    cur_training_ckpt = "dummy"
    for epoch in range(num_epochs):
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        if epoch % log_every_nth == 0:
                print("EPOCH", epoch + 1, "/", 100, "  ...   Mean Loss:", np.mean(mean_losses_train), flush=True)
                
                print("Current Learning Rate:", current_lr)
                losses.append(np.mean(mean_losses_train))
                if np.mean(mean_losses_train) < cur_best_train_loss:
                    print("Saving checkpoint")
                    cur_best_train_loss = np.mean(mean_losses_train)
                    if os.path.isfile(cur_training_ckpt):
                        os.remove(cur_training_ckpt)
                    cur_training_ckpt = save_checkpoint(model, optimizer, epoch, cur_best_train_loss, checkpoint_dir=out_folder, filename=run_token + '_checkpoint_epoch%d' % epoch)
                    # remove old checkpoints
                    
                    
                mean_losses_train = []

        model, optimizer, mean_loss = training_step(epoch, 
                                                    model,
                                                      train_loader,
                                                        use_faces,
                                                        cache_dir,
                                                        optimizer,
                                                        loss_fn,
                                                        mean_losses_train,
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
        cur_best_train_loss = np.mean(mean_losses_train)

        if epoch % validate_every_nth_epoch == 0:
            mean_loss_val = validation_step(epoch,
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
        wandb.log({"Epoch": epoch, "Training Loss": np.mean(mean_losses_train), "Validation Loss": mean_loss_val, "Learning Rate": current_lr})