""" Pytorch lightning module for training DiffusionNet"""
import pytorch_lightning as pl
import torch
from torch.optim import Adam

from membrain_pick.networks.diffusion_net import DiffusionNet
from membrain_pick.optimization.mean_shift_utils import MeanShiftForwarder
from membrain_pick.optimization.mean_shift_losses import MeanShift_loss
from membrain_pick.optimization.optim_utils import weighted_MSELoss, CombinedLoss


class DiffusionNetModule(pl.LightningModule):
    def __init__(self, 
                 C_in, 
                 C_out, 
                 C_width=16, 
                 N_block=6, 
                 mlp_hidden_dims=None, 
                 mean_shift_output=False,
                 mean_shift_bandwidth=7.,
                 mean_shift_max_iter=10,
                 mean_shift_margin=2.,
                 dropout=True, 
                 with_gradient_features=True, 
                 with_gradient_rotations=True, 
                 lstm_first=False, 
                 device="cuda:0", 
                 fixed_time=None, 
                 one_D_conv_first=False, 
                 clamp_diffusion=False, 
                 visualize_diffusion=False, 
                 visualize_grad_rotations=False, 
                 visualize_grad_features=False):
        super().__init__()
        # Initialize the DiffusionNet with the given arguments.
        self.model = DiffusionNet(C_in=C_in, 
                                  C_out=C_out, 
                                  C_width=C_width, 
                                  N_block=N_block, 
                                  mlp_hidden_dims=mlp_hidden_dims, 
                                  dropout=dropout, 
                                  with_gradient_features=with_gradient_features, 
                                  with_gradient_rotations=with_gradient_rotations, 
                                  lstm_first=lstm_first, 
                                  device=device, 
                                  fixed_time=fixed_time, 
                                  one_D_conv_first=one_D_conv_first, 
                                  clamp_diffusion=clamp_diffusion, 
                                  visualize_diffusion=visualize_diffusion, 
                                  visualize_grad_rotations=visualize_grad_rotations, 
                                  visualize_grad_features=visualize_grad_features
                                  )
        self.mean_shift_output = mean_shift_output
        if self.mean_shift_output:
            self.ms_module = MeanShiftForwarder(
                        bandwidth=mean_shift_bandwidth,
                        max_iter=mean_shift_max_iter,
                        margin=mean_shift_margin,
                        device=device,
                    )

    def define_loss(self):
        mse_loss_fn = weighted_MSELoss()
        if self.mean_shift_output:
            self.ms_loss = MeanShift_loss()
        self.criteria = {
            "mse": mse_loss_fn,
            "ms": self.ms_loss
        }
        self.criterion = CombinedLoss(self.criteria)

    def forward(self, batch):
        # Forward pass through DiffusionNet
        features, mass, L, evals, evecs, gradX, gradY, edges, faces, verts_orig = unpack_batch(batch)

        out = {
            "mse": self.model(features, mass, L, evals, evecs, gradX, gradY, edges, faces)
        }
        if self.mean_shift_output:
            out_ms = self.ms_module.mean_shift_for_seeds(
                coords=verts_orig, 
                nn_weights=out["mse"], 
                seeds=verts_orig
                )
            out["ms"] = out_ms
            
        return out

    def configure_optimizers(self):
        # Configure optimizers and schedulers (if needed)
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer
 
    def training_step(self, batch, batch_idx):
        # Unpack data
        targets = {
            "mse": batch["label"],
            "ms": batch["gt_pos"]
        }
        weights = {
            "mse": batch["vert_weights"],
            "ms": torch.ones_like(batch["gt_pos"]) # might need to change this
        }
        
        preds = self(batch)
        
        # Calculate loss
        loss = self.criterion(preds, targets, weights)
        # Log training loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        targets = {
            "mse": batch["label"],
            "ms": batch["gt_pos"]
        }
        weights = {
            "mse": batch["vert_weights"],
            "ms": torch.ones_like(batch["gt_pos"]) # might need to change this
        }
        preds = self(batch)
        loss = self.criterion(preds, targets, weights)
        # Log validation loss
        self.log('val_loss', loss)

    
    def test_step(self, batch, batch_idx):
        targets = {
            "mse": batch["label"],
            "ms": batch["gt_pos"]
        }
        weights = {
            "mse": batch["vert_weights"],
            "ms": torch.ones_like(batch["gt_pos"]) # might need to change this
        }
        preds = self(batch)
        loss = self.criterion(preds, targets, weights)
        # Log test loss
        self.log('test_loss', loss)


def unpack_batch(batch):
    diffusion_inputs = batch["diffusion_inputs"]
    features = diffusion_inputs["features"]
    mass = diffusion_inputs["mass"]
    L = diffusion[inputs]["L"]
    evals = diffusion_inputs["evals"]
    evecs = diffusion_inputs["evecs"]
    gradX = diffusion_inputs["gradX"]
    gradY = diffusion_inputs["gradY"]
    edges = diffusion_inputs["edges"]
    faces = diffusion_inputs["faces"]
    verts_orig = batch["verts_orig"]
    return features, mass, L, evals, evecs, gradX, gradY, edges, faces, verts_orig